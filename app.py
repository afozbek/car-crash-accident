import cv2
import numpy as np

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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Çıktıları denemek isterseniz aşağıdan bakabilirsiniz.
# print(f"{bcolors.FAIL} VIDEO YUKLENIRKEN HATA OLUŞTU 🤕🤕 {bcolors.ENDC}")
# print(f"{bcolors.OKBLUE} SUCCESSFULLY INSTALLED 🤕🤕 {bcolors.ENDC}")
# print(f"{bcolors.OKGREEN} VIDEO YUKLENIRKEN HATA OLUŞTU 🤕🤕 {bcolors.ENDC}")
# print(f"{bcolors.HEADER} VIDEO YUKLENIRKEN HATA OLUŞTU 🤕🤕 {bcolors.ENDC}")
# print(f"{bcolors.WARNING} VIDEO YUKLENIRKEN HATA OLUŞTU 🤕🤕 {bcolors.ENDC}")
# print(f"{bcolors.UNDERLINE} VIDEO YUKLENIRKEN HATA OLUŞTU 🤕🤕 {bcolors.ENDC}")

def useDetections(detections):
  """
  We will implement an algorithm to use this detections dictionary

  and eventually returns if any car crash accident happens

  If happens we will try to inform others that accident happens
  """