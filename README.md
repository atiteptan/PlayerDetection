# Introduction
This is the repository for the trained model in the paper: **Players and Teams Detection in a Soccer Video Using Deep Learning-based Models**.

The 3 trained models can be found in the "Model" folder. Using Yolov5-tph apply the weights listed in the folder on the SoccerTrack dataset.

Please to refer to [Yolov5-tph](https://github.com/cv516Buaa/tph-yolov5) for original model repository and [SoccerTrack](https://github.com/AtomScott/SportsLabKit) for original dataset.

# Installation
Download and install Yolov5-tph and download the SoccerTrack dataset
Please note that you do not need to install SportsLabKit since it may cause conflicts with the packages requirements that YOLO-tph uses, you only need to download the datset from Kaggle.
# Training
Yolov5-tph
``` bash
python train.py --img 1536 --adam --batch 4 --epochs 32 --data /content/data.yaml --weights yolov5m.pt --hy data/hyps/hyp.scratch-med.yaml --cfg Yolov5m-xs-tph.pt --name v5l-xs-tph
```
Yolov5-tph-plus
``` bash
python train.py --img 1536 --adam --batch 4 --epochs 34 --data /content/data.yaml --weights yolov5m.pt --hy data/hyps/hyp.scratch-med.yaml --cfg Yolov5m-tph-plus.pt --name v5l-tph-plus
```
# Detect
``` bash
python detect.py --weights Yolov5m-xs-tph.pt --img 1536 --conf 0.6 --iou-thres 0.7 --source F_20220220_1_1140_1170_Team1Corner.mp4 --save-txt --save-conf
```
# Code Snippets
The following is some sample code snippets you can use to draw the heatmap using the label files from the YOLO models.

## Draw the pitch
Using the python module [mplsoccer](https://github.com/andrewRowlinson/mplsoccer/tree/main) you can easily draw and define the shape of af football pitch.
``` python
import pprint
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from mplsoccer.dimensions import valid

pitch = Pitch(pitch_type='opta', axis=True, label=True, pitch_color='black')
fig, ax = pitch.draw()
ax.plot(0, 100, 'ro', markersize=10) #Draw dots at each corner of the pitch
ax.plot(100, 100, 'ro', markersize=10)
ax.plot(0, 0, 'ro', markersize=10)
ax.plot(100, 0, 'ro', markersize=10)
```
![Empty Pitch](https://github.com/atiteptan/PlayerDetection/blob/main/Empty%20Pitch.png)

## Calculating the Homographic Matrix
Using the 4 corners of the images and pitch we can calculate the homographic matrix to be used to translate the coordinates into top down perspective
``` python
import numpy as np
#Size of the dataset images
x1 = 0
x2 = 6500
x3 = 6500
x4 = 0

y1 = 0
y2 = 0
y3 = 1000
y4 = 1000
#Size of the pitch
u1 = 0
u2 = 100
u3 = 100
u4 = 0

v1 = 0
v2 = 0
v3 = 100
v4 = 100

# Define source points (wide-angle image)
src_points = np.array([
    [x1, y1],  # Point 1 in wide-angle image
    [x2, y2],  # Point 2 in wide-angle image
    [x3, y3],  # Point 3 in wide-angle image
    [x4, y4]   # Point 4 in wide-angle image
], dtype='float32')

# Define destination points (top-down view)
dst_points = np.array([
    [u1, v1],  # Corresponding point 1 in top-down view
    [u2, v2],  # Corresponding point 2 in top-down view
    [u3, v3],  # Corresponding point 3 in top-down view
    [u4, v4]   # Corresponding point 4 in top-down view
], dtype='float32')

import cv2

# Calculate the homography matrix
H, status = cv2.findHomography(src_points, dst_points)

if H is not None:
    print("Homography Matrix:\n", H)
else:
    print("Homography calculation failed.")
```
## Applying the homographic matrix
``` python
def transform_coordinates(points, H):
    transformed_points = []
    for point in points:
        point_homogeneous = np.array([point[0], point[1], 1])  # Convert to homogeneous coordinates
        transformed_point = H @ point_homogeneous              # Apply homography
        transformed_point /= transformed_point[2]              # Normalize to get Cartesian coordinates
        transformed_points.append(transformed_point[:2])       # Append (x, y) only
    return np.array(transformed_points)
x_center1 = 0.665077*6500 #These are sample outputs from the Yolo model
y_center1 = 0.547*1000
x_center2 = 0.629*6500
y_center2 = 0.529*1000
# Example usage: transforming a set of coordinates from wide-angle view
wide_angle_coords = np.array([[x_center1, y_center1], [x_center2, y_center2]])  # Add your coordinates here
top_down_coords = transform_coordinates(wide_angle_coords, H)

print("Transformed Top-Down Coordinates:\n", top_down_coords)
```
