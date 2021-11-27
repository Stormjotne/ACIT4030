"""
Script that uses Matterport's Mask RCNN implementation to infer and output 2D bounding boxes for the classes:
Person, Car and Bicycle
It uses the pre-trained weights for MS COCO dataset with 80 classes.
Some of the outputs are interpreted and altered to fit with the KITTI dataset with three classes.
"""

import os
from pathlib import Path
import sys
import random
import time
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
#  Import Mask RCNN
from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib
#  Import COCO config
import Mask_RCNN.samples.coco.coco as coco

#  Root directory of the project
ROOT_DIR = os.path.abspath("Mask_RCNN/")
#  sys.path.append(ROOT_DIR)  # To find local version of the library
#  sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

#  Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

#  Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
#  Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

#  Directory of images to run detection on
DATASET_DIR = Path("E:/Datasets")
KITTI_2D_STRING = "KITTI/object/training/image_2/"
IMAGE_DIR = Path(DATASET_DIR, KITTI_2D_STRING)
#  Output file folder
OUT_DIR = Path("output")
#  Output file for 2D bounding box data
OUT_FILE = Path(OUT_DIR, "rgb_detection.txt")
MASK_FILE = Path(OUT_DIR, "mask_detection.txt")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
'''
# Load COCO dataset
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR, "train")
dataset.prepare()

# Print class names
print(dataset.class_names)
'''
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

bicycle_id = 2
car_like_ids = [3, 6, 8]
vehicle_ids = range(3, 10)
ignored_vehicle_ids = [4, 5, 7, 9]


def output_2D_bounding_box(name, result):
   """
   Write 2D Bounding Box results to rgb_detection.txt
   Skips and modifies COCO labels to adhere to Frustum PointNets' KITTI labels.
   """
   line_prefix = "dataset/" + KITTI_2D_STRING
   line_image_name = name
   #   Loop through detected objects and create output string for each 2D bounding box.
   for object_index in range(0, len(result['class_ids'])):
      #  print(object_index)
      line_object_id = result['class_ids'][object_index]
      #  Skip object id's 10 and up
      if line_object_id > 9:
         continue
      #  Catch ignored vehicles and skip
      elif line_object_id in ignored_vehicle_ids:
         continue
      #  Catch car-like objects and change ID to 2
      elif line_object_id in vehicle_ids:
         line_detection_score = result['scores'][object_index]
         (start_Y, start_X, end_Y, end_X) = result["rois"][object_index]
         #  Left, Top, Right, Bottom
         line_bounding_box = "{} {} {} {}".format(start_X, start_Y, end_X, end_Y)
         line_out = "{} {} {} {}".format(line_prefix + line_image_name, 2,
                                         line_detection_score, line_bounding_box)
         with OUT_FILE.open("a") as output_file:
            output_file.write(line_out + "\n")
      #  Catch bicycle objects and change ID to 3
      elif line_object_id == bicycle_id:
         line_detection_score = result['scores'][object_index]
         (start_Y, start_X, end_Y, end_X) = result["rois"][object_index]
         #  Left, Top, Right, Bottom
         line_bounding_box = "{} {} {} {}".format(start_X, start_Y, end_X, end_Y)
         line_out = "{} {} {} {}".format(line_prefix + line_image_name, 3,
                                         line_detection_score, line_bounding_box)
         with OUT_FILE.open("a") as output_file:
            output_file.write(line_out + "\n")
      #  Person or Dontcare
      else:
         line_detection_score = result['scores'][object_index]
         (start_Y, start_X, end_Y, end_X) = result["rois"][object_index]
         #  Left, Top, Right, Bottom
         line_bounding_box = "{} {} {} {}".format(start_X, start_Y, end_X, end_Y)
         '''line_bounding_box = "{} {} {} {}".format(result['rois'][object_index][1], result['rois'][object_index][0],
                                                  result['rois'][object_index][3], result['rois'][object_index][2])'''
         line_out = "{} {} {} {}".format(line_prefix + line_image_name, line_object_id,
                                         line_detection_score, line_bounding_box)
         with OUT_FILE.open("a") as output_file:
            output_file.write(line_out + "\n")


'''Configure inference run here'''
total_images = 1
#  Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
batch_size = config.GPU_COUNT * config.IMAGES_PER_GPU
number_of_runs = (total_images - 1) // batch_size + 1
print(number_of_runs)
runs = 0
print("Starting inference process on {} images.".format(total_images))
time.sleep(0.5)
while runs < number_of_runs:
   print("Epoch number: {}".format(runs))
   if batch_size == 1:
       '''Single Image'''
       # Random file
       # chosen_file_name = random.choice(file_names)
       # Specific file name
       chosen_file_name = "007480.png"
       print(chosen_file_name)
       image = skimage.io.imread(os.path.join(IMAGE_DIR, chosen_file_name))
   
       # Run detection on Mask RCNN
       results = model.detect([image], verbose=1)
   
       # Output results
       r = results[0]
       output_2D_bounding_box(chosen_file_name, r)
   else:
       '''Multiple Images'''
       # Random sample
       # chosen_file_names = random.sample(file_names, batch_size)
       # Iterate through folder
       chosen_file_names = file_names[runs * batch_size:(runs + 1) * batch_size]
       print(chosen_file_names)
       images = [skimage.io.imread(os.path.join(IMAGE_DIR, name)) for name in chosen_file_names]
   
       # Run detection
       results = model.detect(images, verbose=1)
   
       # Output results
       r_index = 0
       for r in results:
         output_2D_bounding_box(chosen_file_names[r_index], r)
         r_index += 1
   runs += 1
