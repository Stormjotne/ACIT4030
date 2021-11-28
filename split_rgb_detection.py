"""

"""
from pathlib import Path
import re
r_x = re.compile("(\d{6}).png")
image_set_dir = Path("frustum-pointnets/kitti/image_sets/")
output_dir = Path("output/rgb_detections/")
output_dir.mkdir(parents=True, exist_ok=True)
training_set = Path(image_set_dir, "train.txt")
validation_set = Path(image_set_dir, "val.txt")
rgb_detection_file = Path("output/rgb_detection.txt")
train_split = Path(output_dir, "rgb_detection_train.txt")
train_split.touch(exist_ok=True)
validation_split = Path(output_dir, "rgb_detection_val.txt")
validation_split.touch(exist_ok=True)
print(training_set)
training_file_names = []
validation_file_names = []
#  Read from the Frustum PointNets training split
try:
	with training_set.open('r') as file_name_file:
		for index, line in enumerate(file_name_file):
			training_file_names.append(line.strip())
except IOError:
	print("File not accessible.")
#  Read from the Frustum PointNets validation split
try:
	with validation_set.open('r') as file_name_file:
		for index, line in enumerate(file_name_file):
			validation_file_names.append(line.strip())
except IOError:
	print("File not accessible.")
#  Read from the Mask RCNN 2D bounding box detection data file
try:
	with rgb_detection_file.open('r') as detection_file:
		for index, line in enumerate(detection_file):
			#  Use regular expression to extract filename
			current_file_name = r_x.search(line.strip()).group(1)
			#  print(current_file_name)
			if current_file_name in training_file_names:
				# write to training split file
				with train_split.open('a') as train_file:
					train_file.write(line)
			elif current_file_name in validation_file_names:
				# write to validation split file
				with validation_split.open('a') as val_file:
					val_file.write(line)
except IOError:
	print("File not accessible.")
