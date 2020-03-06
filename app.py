import cv2
import numpy as np
import argparse

# SIMPLE IMAGE
# ./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg

# REAL TIME YOLO
# ./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights

# REAL TIME YOLO MIN
# ./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg bin/yolov3-tiny.weights

# VIDEO FILE
# ./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights <video file>

"""
- Modify your cfg file (e.g. yolov3.cfg), change the 3 classes on line
  - 610,
  - 696,
  - 783
  from 80 to classes
- Change the 3 filters in cfg file on line
  - 603,
  - 689,
  - 776
  from 255 to (classes + 5) x 3
"""

IMAGE_PATH="IMAGE_PATH"
CFG_FILE="CFG_FILE"
WEIGHT_FILE="WEIGHT_FILE"
NAMES_FILE="NAMES_FILE"

# handle command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', required=True, help = 'path to input image')
parser.add_argument('-c', '--config', required=True, help = 'path to yolo config file')
parser.add_argument('-w', '--weights', required=True, help = 'path to yolo pre-trained weights')
parser.add_argument('-n', '--names', required=False, help = 'path to yolo pre-trained weights')

args = vars(parser.parse_args())

options = {
  IMAGE_PATH: args["image"],
  CFG_FILE: args["config"],
  WEIGHT_FILE: args["weights"],
  NAMES_FILE: args["names"],
}

image_path = options[IMAGE_PATH]
cfg_file = options[CFG_FILE]
weight_file = options[WEIGHT_FILE]
names_file = options[NAMES_FILE]

print(image_path)
print(cfg_file)
print(weight_file)

# Load network arch.
# darknet = Darknet(cfg_file)

# Load pre trained weights
# darknet.load_weights(weight_file)

# Load COCO obj classes
# class_names = load_class_names(names_file)
# print(darknet)