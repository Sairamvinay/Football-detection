Guide to using pose estimation code:

Required libraries:

openCv2
numpy

NOTE- Output is arranged in the following order:
['Head', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank']

poseEstimation.py takes in, as command line arguments:
--device (CPU or GPU)
--image_file (defunct)
--video_file path to the desired video in any format

The workflow for using this file is as follows:
1. A frame will appear on the video screen with a single key point attached to it. It is your job to either:
    a. Accept the automatic key point candidate using the y key
    b. Reject the automatic key point candidate using any key not in use. The algorithm will go to it's second best guess. This is useful for when you have good quality data, but the algorithm chooses the kicker rather than the goalie.
    c. Click your own key point - click the screen where the key point should be AND THEN use the d key. This is useful when the frame is blurry and it can't find the goalie easily, or, for whatever reason, it want's to pick someone from the crowd. Some blurry frames require you to do this a lot, so make sure to consult the key point chart to find the proper ordering. 
    d. Fix a mistake - go back to the previous key point, delete it, and fix a mistake using the b key.
    e. Skip a key point - not recommended. q key.

2. After you have selected a key point, the algorithm will estimate the next key point. The key point is partially based on the position of the previous key point, so it is more likely to find the goalie the second time around.

3. When all key points are detected, it will move to the next frame. By default, it will find 10 frames for you, uniformly spaced out in the video.

4. After you complete the 10 frames, the script terminates and output is saved as [filename without extension].npy, and a JSON.