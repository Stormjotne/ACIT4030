# ACIT4030
Machine Learning for Images and 3D Data

##  Research project on the use of ML/DL on 3D data and RGB images in Autonomous Driving
Authors: Rashmi Naik and Ruben Jahren

##  Content
This repository includes two submodules that are forked from Matterports implementation of Mask RCNN and Charles Qi's implementation of Frustum PointNets.
Some of the code has been changed and adapted for our project.
We've added new 2D bounding box outputs from Mask RCNN to the Frustum PointNets submodule.
We've also added scripts for setting up the project and running it on an external GPU cluster in collaboration with Simula.
Due to time-constraints, we got stuck on the last part right before submission.

##  Files that are strictly our contribution and/or continuation
### Our code
inference_2dbb.py

split_rgb_detection.py

Mask_RCNN/samples/infer_KITTI.ipynb

### Our outputs
rgb_detections/rgb_detection_train.txt

ACIT4030/output/rgb_detections/rgb_detection_train.txt

ACIT4030/output/rgb_detections/rgb_detection_val.txt
