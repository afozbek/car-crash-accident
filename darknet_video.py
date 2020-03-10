from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

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
        print("Error Happened in ValidateYOLo")
        return

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

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    # Start the big YOLO record loop
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet.network_width(netMain), darknet.network_height(netMain)), interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1 / (time.time() - prev_time))
        cv2.imshow('Car Accident', image)

        # If 'esc' or 'q' key pressed break the loop
        k = cv2.waitKey(2) & 0xFF
        if k == 27 or k == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO(screenRecord=True)
