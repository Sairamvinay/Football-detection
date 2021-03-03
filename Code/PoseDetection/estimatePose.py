import cv2
import time
import numpy as np
import math
import argparse
import json
import os
from random import randint
from copy import deepcopy

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument("--image_file", default="group.jpg", help="Input image")
parser.add_argument("--video_file", default="sample_video.mp4", help="Input Video")

args = parser.parse_args()

protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
nPoints = 18

# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

colors = [ [0,100,255], [0,255,255], [0,100,255], [255,0,0], [0,255,255], [0,100,255],
         [255,0,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,255, 0], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]


def getKeypoints(probMap, threshold=0.1):

    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []

    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


# Find valid connections between the different joints of a all persons present
def getValidPairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs

# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints

def prune_candidates(candidates, last_keypoint, w):
    if last_keypoint == []:
        return candidates
    distances = []
    for candidate in candidates:
        distances.append(math.sqrt((last_keypoint[0]-candidate[0])**2+(last_keypoint[1]-candidate[1])**2))
    '''new_candidates = []
    new_distances = []
    for i, d in enumerate(distances):
        if d/w <= 0.15:
            new_candidates.append(candidates[i])
            new_distances.append(d)'''
    try:
        distances, candidates = zip(*sorted(zip(distances, candidates)))
    except:
        return [(-1,-1)]
    return candidates

t = time.time()
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

# Fix the input Height and get the width according to the Aspect Ratio
'''inHeight = 368
inWidth = int((inHeight/frameHeight)*frameWidth)

inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)
output = net.forward()
print("Time Taken in forward pass = {}".format(time.time() - t))

detected_keypoints = []
keypoints_list = np.zeros((0,3))
keypoint_id = 0
threshold = 0.1'''

input_source = args.video_file
cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()

xd, yd = -1, -1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global xd, yd
    if event == cv2.EVENT_LBUTTONDOWN:
        xd, yd = x, y

cv2.namedWindow('Keypoints')
cv2.setMouseCallback('Keypoints',draw_circle)

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

inHeight = 368
inWidth = int((inHeight/frameHeight)*frameWidth)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
desired_frames = 10
chunck_size = int(total_frames/desired_frames)
frame_counter = chunck_size
point_counter = -1
previousKeypoint = []
c = 0
# Output in format n*m*2 with n frames, m key points, and an x,y position
accepted_points = np.zeros((desired_frames, 13, 2))

while point_counter < desired_frames-1:
    if frame_counter != chunck_size:
        c += 1
        frame_counter += 1
        hasFrame, frame = cap.read()
        continue

    frame_counter = 1
    point_counter += 1
    c += 1

    t = time.time()
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        print(c)
        cv2.waitKey()
        break
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]

    keypoint_id = 0
    keypoints_list = np.zeros((0,3))
    detected_keypoints = []

    for part in range(nPoints):
        probMap = output[0,part,:,:]
        probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
        keypoints = getKeypoints(probMap)
        #print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)

    last_hit = 0

    frameClone = frame.copy()
    imageCache = [frameClone.copy()]
    to_remove = [1, 14, 15, 16, 17]
    new_points = []
    for i, point in enumerate(detected_keypoints):
        if i in to_remove:
            pass
        else:
            new_points.append(detected_keypoints[i])
    detected_keypoints = new_points
    i = 0
    while i<len(detected_keypoints):
        j = 0
        #order[0], order[last_hit] = order[last_hit], order[0]
        candidates = deepcopy(detected_keypoints[i])
        candidates = prune_candidates(candidates, previousKeypoint, frameWidth)
        order = list(range(len(candidates)))
        while j < len(candidates):
            index = order[j]
            cv2.circle(frameClone, (candidates[index][0:2]), 8, colors[i], -1, cv2.LINE_AA)
            cv2.imshow("Keypoints",frameClone)
            cont = cv2.waitKey(0)
            # y - accepted
            if cont == 121:
                imageCache.append(frameClone.copy())
                last_hit = index
                previousKeypoint = candidates[index][0:2]
                j += 1
                accepted_points[point_counter, i, 0] = (candidates[index][0] - frameWidth/2)/frameWidth
                accepted_points[point_counter, i, 1] = (candidates[index][1] - frameHeight/2)/frameHeight
                break
            # d - draw
            elif cont == 100:
                frameClone = imageCache[-1].copy()
                cv2.circle(frameClone, (xd,yd), 8, colors[i], -1, cv2.LINE_AA)
                imageCache.append(frameClone.copy())
                last_hit = index
                previousKeypoint = [xd,yd]
                j += 1
                accepted_points[point_counter, i, 0] = (xd - frameWidth/2)/frameWidth
                accepted_points[point_counter, i, 1] = (yd - frameHeight/2)/frameHeight
                break
            # b - back
            elif cont == 98:
                if i > 0:
                    frameClone = imageCache[-2].copy()
                    i -= 2
                    imageCache.pop(-1)
                    break
            # q - pass keypoint
            elif cont == 113:
                frameClone = imageCache[-1].copy()
                break
            # otherwise - reject
            else:
                frameClone = imageCache[-1].copy()
                j += 1
                continue
        i += 1

np.save(os.path.basename(input_source).split('.')[0], accepted_points)
keypointsMapping = ['Head', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank']

d = {}
for i, point in enumerate(keypointsMapping):
    d[point] = accepted_points[:,i,:].tolist()

json = json.dumps(d)
f = open(os.path.basename(input_source).split('.')[0] + ".json","w")
f.write(json)
f.close()