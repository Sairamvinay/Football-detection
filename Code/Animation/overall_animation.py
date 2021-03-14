import numpy as np
import vtk
from time import sleep
import cv2
import sys


ACTORS = []
def get_sizes(video):

	vcap = cv2.VideoCapture(video)
	if vcap.isOpened(): 
		# get vcap property 
		width  = vcap.get(3)  # float `width`
		height = vcap.get(4)  # float `height`

	vcap.release()
	return (width,height)


def plotLine(start, end, scalex = 10, scaley=7.5):
	s = start[0:3]
	e = end[0:3]
	source = vtk.vtkLineSource()
	source.SetPoint1(s[0]*scalex, s[1]*scaley, s[2])
	source.SetPoint2(e[0]*scalex, e[1]*scaley, e[2])
	mapper = vtk.vtkPolyDataMapper()
	mapper.SetInputConnection(source.GetOutputPort())
	# actor
	actor = vtk.vtkActor()
	actor.SetMapper(mapper)
	ren.AddActor(actor)
	return actor

def clearOldActors():
	if len(ACTORS) != 0:
		for i in range(len(ACTORS)):
			ren.RemoveActor(ACTORS[i])

	else:
		pass

def drawJointLines(joint_pos_final):

	clearOldActors() #remove all the old lines ahead of time in case if any so that the motion of the goalie looks clean

	ACTORS.append(plotLine(joint_pos_final[0],joint_pos_final[1],scalex=1,scaley=1)) # Head -> Right shoulder
	ACTORS.append(plotLine(joint_pos_final[0],joint_pos_final[4],scalex=1,scaley=1)) # Head -> Left shoulder
	ACTORS.append(plotLine(joint_pos_final[0],joint_pos_final[7],scalex=1,scaley=1)) # Head -> Right Hip
	ACTORS.append(plotLine(joint_pos_final[0],joint_pos_final[10],scalex=1,scaley=1)) # Head -> Left Hip

	ACTORS.append(plotLine(joint_pos_final[1],joint_pos_final[2],scalex=1,scaley=1)) # Right Shoulder -> Right elbow
	ACTORS.append(plotLine(joint_pos_final[2],joint_pos_final[3],scalex=1,scaley=1)) # Right Elbow -> Right Wrist

	ACTORS.append(plotLine(joint_pos_final[4],joint_pos_final[5],scalex=1,scaley=1)) # Left Shoulder -> Left elbow
	ACTORS.append(plotLine(joint_pos_final[5],joint_pos_final[6],scalex=1,scaley=1)) # Left Elbow -> Left Wrist

	ACTORS.append(plotLine(joint_pos_final[7],joint_pos_final[8],scalex=1,scaley=1)) # Right Hip -> Right Knee
	ACTORS.append(plotLine(joint_pos_final[8],joint_pos_final[9],scalex=1,scaley=1)) # Right Knee -> Right Ankle

	ACTORS.append(plotLine(joint_pos_final[10],joint_pos_final[11],scalex=1,scaley=1)) # Left Hip -> Left Knee
	ACTORS.append(plotLine(joint_pos_final[11],joint_pos_final[12],scalex=1,scaley=1)) # Left Knee -> Left Ankle

def drawGoalPost(gp_positions):
	pass


BALL_RAD = 0.8

MAGNIFY = 25.0
MAGNIFY_BALL = 10.0
MAGNIFY_BALL_Y = 2.0

WINDOW_WT = 2500
WINDOW_HT = 1200

#OFFSET = 

ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(WINDOW_WT,WINDOW_HT)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.Initialize()

def animate(skeleton, ball, video):

	colors = [ [0,100,255], [0,255,255], [0,100,255], [255,0,0], [0,255,255], [0,100,255],
			[255,0,0], [255,200,100], [255,0,255], [0,255,255], [255,200,100], [255,0,255],
			[0,255,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,200]]

	ball_color = [255,255,255]#[255,110,180]

	spheres = []

	arr = skeleton
	ball_arr = ball
	video_width, video_height = get_sizes(video)

	print("Video size: ",video_width,',',video_height)

	# get the actual video's frame size (H,W)
	# for that Render window's size: (H',W')
	# get the ht scale h = H'/H ; width scale w = W'/W
	# x' = x * w  ; y' = y * h for every joint and the ball
	# we also apply Magnification for every joint and ball seperately

	ratio_wt = WINDOW_WT / video_width
	ratio_ht = WINDOW_HT / video_height

	X_OFFSET = 5
	Y_OFFSET = 3.5#3
	BRING_GOAL_DOWN = 1.5

	# # this creates a 6x4 goal post
	
	# plotLine([-3,-2,0],[-3,2,0]) #vertical left line
	# plotLine([-3,2,0],[3,2,0]) #horizontal line
	# plotLine([3,-2,0],[3,2,0]) #vertical right 

	# LSHOULDER (X1,Y1) {Index= 4}; RSHOULDER: (X2,Y2) {Index= 4} from first frame
	# Mostly Y1= Y2
	
	x1,y1 = arr[0][4] #L-sho
	x2,y2 = arr[0][1] #R-sho

	mapper_joint = ['Head', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank']

	print("The shoulders are at same height: ",(y1==y2))
	print("The width across shoulders: ",(x2-x1))

	#A -> Left bottom coordinate
	#B -> right bottom coordinate
	#C -> left top coordinate
	#D -> right top coordinate

	#Need to draw out: AC (V); BD (V); and CD (H)

	pointA = ((x1 - X_OFFSET), (y1 - BRING_GOAL_DOWN) , 0)
	pointB = ((x2 + X_OFFSET), (y2 - BRING_GOAL_DOWN) , 0)
	pointC = ((x1 - X_OFFSET), (y1 + Y_OFFSET) , 0)
	pointD = ((x2 + X_OFFSET), (y2 + Y_OFFSET) , 0)

	plotLine(pointA,pointC,scalex=6,scaley=5)
	plotLine(pointB,pointD,scalex=6,scaley=5)
	plotLine(pointC,pointD,scalex=6,scaley=5)


	for time_iter,timestep in enumerate(arr[:]):
		
		joint_pos_final = []

		for i,joint in enumerate(timestep):

			x_orig = joint[0]
			y_orig = joint[1]

			x = ratio_wt * x_orig * MAGNIFY
			y = -1 * ratio_ht * y_orig * MAGNIFY #need to invert since the y is always set negative or we see an inverted goalie
			#print("Joint ",mapper_joint[i]," Positions x: ",x," y: ",y, ' color: ',colors[i])

			joint_pos_final.append((x,y,0)) #get the positions of the joints

			#if i == len(spheres):
			#add the first frame neatly alone
			if time_iter == 0:
				
				source = vtk.vtkSphereSource()
				source.SetCenter(x,y,0)
				source.SetRadius(0.5)
				source.Update()
				spheres.append(source)

				# mapper
				mapper = vtk.vtkPolyDataMapper()
				if vtk.VTK_MAJOR_VERSION <= 5:
				    mapper.SetInput(source.GetOutput())
				else:
				    mapper.SetInputConnection(source.GetOutputPort())

				# actor
				actor = vtk.vtkActor()
				actor.GetProperty().SetColor(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255)
				actor.SetMapper(mapper)
				ren.AddActor(actor)

				#we need to re-add 
				if i == 12:
					source = vtk.vtkSphereSource()
					source.SetCenter(x,y,0)
					source.SetRadius(0.5)
					source.Update()
					spheres.append(source)

					# mapper
					mapper = vtk.vtkPolyDataMapper()
					if vtk.VTK_MAJOR_VERSION <= 5:
					    mapper.SetInput(source.GetOutput())
					else:
					    mapper.SetInputConnection(source.GetOutputPort())

					# actor
					actor = vtk.vtkActor()
					actor.GetProperty().SetColor(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255)
					actor.SetMapper(mapper)
					ren.AddActor(actor)

			
			else:
				spheres[i].SetCenter(x,y,0)


		# print("After adding: spheres: ",len(spheres))		

		drawJointLines(joint_pos_final)

		# if time_iter != 0:
		# 	x_ball_orig,y_ball_orig = ball_arr[time_iter]

		# else:
		# 	x_ball_orig,_ = arr[0][0] #fix the head's coordinates as for the ball
		# 	y_ball_orig = ball_arr[time_iter][1] #the ball's y stays the same

		x_ball_orig,y_ball_orig = ball_arr[time_iter]
		
		print("TIMESTEP ",time_iter)
		
		print("BEFORE SCALE")
		print("BALL X:",x_ball_orig)
		print("BALL Y:",y_ball_orig)


		x_ball = x_ball_orig * ratio_wt * MAGNIFY_BALL
		if (time_iter < 2):
			y_ball = y_ball_orig * ratio_ht * MAGNIFY_BALL_Y
		else:
			y_ball = abs(y_ball_orig * ratio_ht * MAGNIFY_BALL)

		print("AFTER SCALE")
		print("BALL X:",x_ball)
		print("BALL Y:",y_ball)
		print('-'*50)
		
		source = vtk.vtkSphereSource()
		source.SetCenter(x_ball,y_ball,0)
		source.SetRadius(BALL_RAD)
		spheres.append(source)
		# mapper
		mapper = vtk.vtkPolyDataMapper()
		if vtk.VTK_MAJOR_VERSION <= 5:
		    mapper.SetInput(source.GetOutput())
		else:
		    mapper.SetInputConnection(source.GetOutputPort())

		# actor
		actor.GetProperty().SetColor(ball_color[0]/255, ball_color[1]/255, ball_color[2]/255)
		actor.SetMapper(mapper)
		ren.AddActor(actor)

		# enable user interface interactor
		renWin.Render()
		iren.Render()
		print("Enter to continue: ")
		input()


def main():

	npy_joint = "FACup_Hit_3.npy"
	npy_ball = "scaled_FACup_Hit_3.npy"
	video_file = "videos/FACup_Hit_3.mp4"

	joint = np.load(npy_joint)
	ball = np.load(npy_ball)

	print(joint[:][:-1].shape, ' is the original shape of the joint')
	animate(joint[:][:-1],ball[:-1],video_file)


main()



