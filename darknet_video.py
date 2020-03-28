from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

import argparse
from pathlib import Path

from app import useDetections, bcolors
from deepcrash.centroidtracker import CentroidTracker

netMain = None
metaMain = None
altNames = None

def validateYOLO(configPath, weightPath, metaPath):
    """
    Validates the YOLO paths and throws errors if error exists

    Parameters
    ----------------
    configPath: str
        Path to the config to evaluate. Raises ValueError if not found

    weightPath: str
        Path to the weight to evaluate. Raises ValueError if not found

    weightPath: str
        Path to the meta to evaluate. Raises ValueError if not found
    """
    global metaMain, netMain, altNames
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" + os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" + os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" + os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

def YOLO(
    videoPath="./data/videos/car-accident-2.mp4",
    configPath = "./cfg/yolov3.cfg",
    weightPath = "./yolov3.weights",
    metaPath = "./cfg/coco-new.data",
    screenRecord=False
):
    try:
        validateYOLO(configPath, weightPath, metaPath)
    except Exception:
        print("Error Happened in ValidateYOLO")
        return

    centroidTracker = CentroidTracker(50)

    # Select a video element to apply YOLO
    cap = cv2.VideoCapture(videoPath)
    if screenRecord:
        cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # Write YOLO results to an avi video file
    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    # Prepare video
    print("Starting the YOLO loop...")

    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)

        # Resized borderless video frame
        frame_resized = cv2.resize(frame_rgb, (darknet.network_width(netMain), darknet.network_height(netMain)), interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        # We have now detections dictionary
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

        # Inside app.py
        # Main App Function
        useDetections(detections, frame_resized, centroidTracker)

        print(1 / (time.time() - prev_time))

        # If 'esc' or 'q' key pressed break the loop
        k = cv2.waitKey(2) & 0xFF
        if k == 27 or k == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
    out.release()

if __name__ == "__main__":
    # handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=Path, required=False, default=Path(__file__).absolute().parent / "data/videos", help = 'path to default video directory')
    parser.add_argument('-v', '--video', type=Path, required=False, default = "car-accident-2.mp4", help = 'path to video file')

    parser.add_argument('--rec', dest='record', action='store_true')
    parser.add_argument('--no-rec', dest='record', action='store_false')
    parser.set_defaults(record=True)

    args = vars(parser.parse_args())

    dirPath = args["directory"]
    videoPath = args["video"]
    totalVideoPath = args["directory"] / args["video"]

    if os.path.exists(totalVideoPath):
        print(f"{bcolors.OKGREEN}\t MESSAGE: Starting the YOLO func 🤘🤘.{bcolors.ENDC}")

        YOLO(videoPath=str(totalVideoPath), screenRecord=args["record"])
    else:
        print(f"{bcolors.FAIL}\t VIDEO YUKLENIRKEN HATA OLUŞTU 🤕🤕 {bcolors.ENDC}")
        print("The video path is not correct. Please check the video file path \n")

    print(f"{bcolors.OKGREEN}\t YOLO BAŞARI İLE SONLANDIRILDI! TEBRİK EDERİM 😏😇 {bcolors.ENDC}")