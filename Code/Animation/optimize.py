#from final_animation import animate
from overall_animation import animate
import numpy as np
import math
from scipy.interpolate import interp1d

np.seterr(divide='raise', invalid='raise')

# Interpolate a point
def interpolate_point(goal, start, prev, point):
	goalxx, goalyx, goalxy, goalyy = goal[2], goal[2], goal[0], goal[1]
	startxx, startyx, startxy, startyy = start[2], start[2], start[0], start[1]
	prevxx, prevyx, prevxy, prevyy = prev[2], prev[2], prev[0], prev[1]
	pointxx, pointyx, pointxy, pointyy = point[2], point[2], point[0], point[1]
	'''return (interpolate(goalxx, startxx, prevxx, pointxx, goalxy, startxy, prevxy, pointxy),
		interpolate(goalyx, startyx, prevyx, pointyx, goalyy, startyy, prevyy, pointyy))
	'''
	#input([goalxx, startxx, prevxx, pointxx])
	xinter = interp1d(np.asarray([goalxx, startxx, prevxx]), np.asarray([goalxy, startxy, prevxy]), kind='quadratic', fill_value="extrapolate")
	yinter = interp1d(np.asarray([goalyx, startyx, prevyx]), np.asarray([goalyy, startyy, prevyy]), kind='quadratic', fill_value="extrapolate")
	return xinter(np.asarray(pointxx)), yinter(np.asarray(pointyx))

# Helper function. Calculates the interpolation
'''def interpolate(goal, start, prev, point):
	try:
		t1 = ((point-prev)*(point-goal)*start)/((start-prev)*(start-goal))
		t2 = ((point-start)*(point-goal)*prev)/((prev-start)*(prev-goal))
		t3 = ((point-start)*(point-prev)*goal)/((goal-start)*(goal-prev))
	except:
		if start == prev or start == goal:
			start += 0.001
		if goal == prev:
			prev += 0.001
		return interpolate(goal, start, prev, point)
	return t1+t2+t3'''

def interpolate(goalx, startx, prevx, pointx, goaly, starty, prevy, pointy):
	try:
		t1 = ((pointx-prevx)*(pointx-goalx)*starty)/((startx-prevx)*(startx-goalx))
		t2 = ((pointx-startx)*(pointx-goalx)*prevy)/((prevx-startx)*(prevx-goalx))
		t3 = ((pointx-startx)*(pointx-prevx)*goaly)/((goalx-startx)*(goalx-prevx))
	except:
		if startx == prevx or startx == goalx:
			print(startx)
			print(prevx)
			input(goalx)
			startx += 0.001
		if goalx == prevx:
			print(startx)
			print(prevx)
			input(goalx)
			prevx += 0.001
		return interpolate(goaly, starty, prevy, pointy, goalx, startx, prevx, pointx)
	return t1+t2+t3

# Find closest joint
def find_cloest(skeleton, ball):
	d = math.inf
	j = -1
	for joint in skeleton:
		dc = np.linalg.norm(np.asarray(joint) - np.asarray(ball))
		if dc < d:
			d = dc
			j = joint
	return j

#Create final frame
def create_virtual_goal(skeleton, ball):
	#ball = np.asarray([0.6, 0])
	cloest_joint = find_cloest(skeleton, ball)
	# One positive, other negetive, do a mirror flip around hit
	if cloest_joint[0]*ball[0] < 0:
		skeleton = flip(skeleton)
		cloest_joint = find_cloest(skeleton, ball)
	# Find translation
	translation = ball - cloest_joint
	# Apply translation
	for i in range(skeleton.shape[0]):
		skeleton[i] += translation
	return skeleton

# Flip a skeleton around it's hip
def flip(skeleton):
	xc, yc = skeleton[7,0], skeleton[7,1]
	for i, joint in enumerate(skeleton):
		x = joint[0]
		y = joint[1]
		x_prime = (x-xc)*math.cos(math.pi) - (y-yc)*math.sin(math.pi) + xc
		y_prime = (x-xc)*math.sin(math.pi) + (y-yc)*math.cos(math.pi) + yc
		skeleton[i, 0] = x_prime
		skeleton[i, 1] = y_prime
	return skeleton

#Perform point-wise interpolation across the time series
def optimize(arr, ball, t):
	goal = create_virtual_goal(arr[t], ball[t])
	start = arr[0]
	arr[t] = goal
	arr = arr[::-1]
	newArr = []
	for i, skeleton in enumerate(arr):
		for j in range(skeleton.shape[0]):
			joint = skeleton[j]
			if i == len(arr)-t-1:
				continue
			prev_ind = i+1
			next_ind = i-1
			if prev_ind == len(arr)-t-1:
				prev_ind += 1
			if prev_ind == len(arr):
				prev_ind -= 3
			if next_ind == len(arr)-t-1:
				next_ind -=1
			if next_ind < 0:
				next_ind += 3
			prev = arr[prev_ind][j]
			nxt = arr[next_ind][j]
			skeleton[j] = interpolate_point(goal[j].tolist()+[len(arr)-1-t], prev.tolist()+[prev_ind], nxt.tolist()+[next_ind], joint.tolist()+[i])
		newArr.append(skeleton)
	newArr[0] = (start)
	#newArr.insert(t, goal)
	newArr = np.asarray(newArr)[::-1]
	return newArr

def main(skeleton_filename, ball_filename, video, t):
	skeleton = np.load(skeleton_filename+'.npy')
	ball = np.load(ball_filename+'.npy')
	animate(skeleton, ball, video)
	for i in range(10):
		skeleton = optimize(skeleton, ball, t)
		'''animate(skeleton, ball, video)
		cont = input('Continue?')
		if 'n' in cont.lower():
			break'''
	animate(skeleton[0:-1], ball[0:-1], video)

main('school_hit_35', 'scaled_school_hit_35', 'school_hit_53.mov',4)

'''arr = np.load('school_hit_35.npy')
arr = create_virtual_goal(arr[8], np.asarray([0.5,0.5]))
from matplotlib import pyplot as plt

plt.plot(arr[:,0],arr[:,1], 'ob')
plt.show()'''











