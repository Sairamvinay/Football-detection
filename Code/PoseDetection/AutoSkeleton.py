import os

# Folder name here
folder = ''

movies = os.listdir(folder)

for movie in movies:
	input(os.path.join(folder, movie))
	os.system('python3 estimatePose.py --video_file ' + os.path.join(folder, movie))