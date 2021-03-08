import numpy as np
import vtk
import argparse
from time import sleep

parser = argparse.ArgumentParser(description='Animate skeleton from array')
parser.add_argument("--np_file", default="sample_video.mp4", help="Input Video")

colors = [ [0,100,255], [0,255,255], [0,100,255], [255,0,0], [0,255,255], [0,100,255],
		[255,0,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
		[0,255, 0], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]


ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(5000, 1000)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.Initialize()

args = parser.parse_args()

arr = np.load(args.np_file+'.npy').tolist()

spheres = []

for timestep in arr[0:-1]:
	for i,joint in enumerate(timestep):
		x = -1*joint[0]*50
		y = -1*joint[1]*50
		if i == len(spheres):
			source = vtk.vtkSphereSource()
			source.SetCenter(x,y,0)
			source.SetRadius(0.5)
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
	# enable user interface interactor
	renWin.Render()
	iren.Render()
	input()
