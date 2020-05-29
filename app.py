import cv2
import numpy as np

from deepcrash.centroidtracker import CentroidTracker
import argparse
import time

# Çıktıları denemek isterseniz aşağıdan bakabilirsiniz.
# print(f"{bcolors.FAIL} VIDEO YUKLENIRKEN HATA OLUŞTU 🤕🤕 {bcolors.ENDC}")
# print(f"{bcolors.OKBLUE} SUCCESSFULLY INSTALLED 🤕🤕 {bcolors.ENDC}")
# print(f"{bcolors.OKGREEN} VIDEO YUKLENIRKEN HATA OLUŞTU 🤕🤕 {bcolors.ENDC}")
# print(f"{bcolors.HEADER} VIDEO YUKLENIRKEN HATA OLUŞTU 🤕🤕 {bcolors.ENDC}")
# print(f"{bcolors.WARNING} VIDEO YUKLENIRKEN HATA OLUŞTU 🤕🤕 {bcolors.ENDC}")
# print(f"{bcolors.UNDERLINE} VIDEO YUKLENIRKEN HATA OLUŞTU 🤕🤕 {bcolors.ENDC}")

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def useDetections(detections, img, centroidTracker):
    """
    We will implement an algorithm to use this detections dictionary
    and eventually returns if any car crash accident happens
    If happens we will try to inform others that accident happens
    """

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    rects = draw_rectangles(detections, img)

    (is_accident_happened, box) = find_accidents(rects)

    objects = centroidTracker.update(rects)

    img = draw_circles(objects, img)

    if is_accident_happened:
        # Kaza oluştu ise ekrana çiz
        draw_errors(img, box)

    cv2.imshow("Car Accident", img)

    import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.show()

def find_accidents(rects):
    is_accident_happen = False
    box = (0,0,0,0)

    for i in range(len(rects)):
        # A_xmin, A_xmax, A_ymin, A_ymax = rects[i]
        A_xmin, A_ymin, A_xmax, A_ymax = rects[i]
        for j in range(len(rects)):
            # B_xmin, B_xmax, B_ymin, B_ymax = rects[j]
            B_xmin, B_ymin, B_xmax, B_ymax = rects[j]

            if A_xmin == B_xmin and A_xmax == B_xmax and A_ymin == B_ymin and A_ymax == B_ymax:
                # print("\nEşit olduğundan devam ediliyor\n")
                continue

            print("\nLOGLAMA\n")
            print("BOX-A: ", str(rects[i]))
            print("BOX-B: ", str(rects[j]))
            
            #if (A_xmin < B_xmin and B_xmin < A_xmax and B_xmax > A_xmax) and ((A_ymin < B_ymin and B_ymin < A_ymax ) or (B_ymin < A_ymin and A_ymin < B_ymax) ) or (A_ymin < B_ymin and B_ymin < A_ymax and B_ymax > A_ymax):

            if (A_xmin < B_xmin and B_xmin < A_xmax and B_xmax > A_xmax) and (A_ymin < B_ymin and B_ymin < A_ymax and B_ymax > A_ymax): # 7
                print("\n\n\n")
                print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
                print("\n\n\n")
                is_accident_happen = True
                box = rects[i]
            elif (B_xmin < A_xmin and A_xmin < B_xmax and A_xmax > B_xmax) and (B_ymin < A_ymin and A_ymin < B_ymax and A_ymax > B_ymax): # 1
                print("\n\n\n")
                print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
                print("\n\n\n")
                is_accident_happen = True
                box = rects[i]

            elif (B_xmin < A_xmin and A_xmin < B_xmax and A_xmax > B_xmax) and (A_ymin < B_ymin and B_ymin < B_ymax and A_ymax < B_ymax): # 2
                print("\n\n\n")
                print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
                print("\n\n\n")
                is_accident_happen = True
                box = rects[i]

            elif (B_xmin < A_xmin and A_xmin < B_xmax and A_xmax > B_xmax) and (A_ymin < B_ymin and B_ymin < A_ymax and B_ymax > A_ymax): # 3
                print("\n\n\n")
                print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
                print("\n\n\n")
                is_accident_happen = True
                box = rects[i]

            elif (A_xmin < B_xmin and B_xmin < A_xmax and B_xmax > A_xmax) and (B_ymin < A_ymin and A_ymin < B_ymax and A_ymax > B_ymax): # 5  
                print("\n\n\n")
                print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
                print("\n\n\n")
                is_accident_happen = True
                box = rects[i]

            elif (A_xmin < B_xmin and B_xmin < A_xmax and B_xmax > A_xmax) and (A_ymin < B_ymin and B_ymin < B_ymax and A_ymax > B_ymax): # 6
                print("\n\n\n")
                print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
                print("\n\n\n")
                is_accident_happen = True
                box = rects[i]

            elif (A_xmin < B_xmin and B_xmin < B_xmax and A_xmax > B_xmax) and (A_ymin < B_ymin and B_ymin < A_ymax and B_ymax > A_ymax): # 4 
                print("\n\n\n")
                print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
                print("\n\n\n")
                is_accident_happen = True
                box = rects[i]

            # elif (A_xmin < B_xmin and B_xmin < B_xmax and A_xmax > B_xmax) and (B_ymin < A_ymin and A_ymin < B_ymax and A_ymax > B_ymax):  # 8
            #     print("\n\n\n")
            #     print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
            #     print("\n\n\n")
            #     is_accident_happen = True
            #     box = rects[i]

            elif (A_xmin < B_xmin and B_xmin < B_xmax and A_xmax > B_xmax) and (A_ymin < B_ymin and B_ymin < A_ymax and B_ymax > A_ymax): # 9
                print("\n\n\n")
                print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
                print("\n\n\n")
                is_accident_happen = True
                box = rects[i]


            # A ve B ler yer değiştirir: 

            elif (B_xmin < A_xmin and A_xmin < B_xmax and A_xmax > B_xmax) and (B_ymin < A_ymin and A_ymin < B_ymax and A_ymax > B_ymax): # 7
                print("\n\n\n")
                print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
                print("\n\n\n")
                is_accident_happen = True
                box = rects[i]
            
            elif (A_xmin < B_xmin and B_xmin < A_xmax and B_xmax > A_xmax) and (A_ymin < B_ymin and B_ymin < A_ymax and B_ymax > A_ymax): # 1
                print("\n\n\n")
                print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
                print("\n\n\n")
                is_accident_happen = True
                box = rects[i]

            elif (A_xmin < B_xmin and B_xmin < A_xmax and B_xmax > A_xmax) and (B_ymin < A_ymin and A_ymin < A_ymax and B_ymax < A_ymax): # 2
                print("\n\n\n")
                print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
                print("\n\n\n")
                is_accident_happen = True
                box = rects[i]

            elif (A_xmin < B_xmin and B_xmin < A_xmax and B_xmax > A_xmax) and (B_ymin < A_ymin and A_ymin < B_ymax and A_ymax > B_ymax): # 3
                print("\n\n\n")
                print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
                print("\n\n\n")
                is_accident_happen = True
                box = rects[i]


            elif (B_xmin < A_xmin and A_xmin < B_xmax and A_xmax > B_xmax) and (A_ymin < B_ymin and B_ymin < A_ymax and B_ymax > A_ymax): # 5  
                print("\n\n\n")
                print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
                print("\n\n\n")
                is_accident_happen = True
                box = rects[i]


            elif (B_xmin < A_xmin and A_xmin < B_xmax and A_xmax > B_xmax) and (B_ymin < A_ymin and A_ymin < A_ymax and B_ymax > A_ymax): # 6
                print("\n\n\n")
                print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
                print("\n\n\n")
                is_accident_happen = True
                box = rects[i]

            elif (B_xmin < A_xmin and A_xmin < A_xmax and B_xmax > A_xmax) and (B_ymin < A_ymin and A_ymin < B_ymax and A_ymax > B_ymax): # 4 
                print("\n\n\n")
                print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
                print("\n\n\n")
                is_accident_happen = True
                box = rects[i]

            # elif (B_xmin < A_xmin and A_xmin < A_xmax and B_xmax > A_xmax) and (A_ymin < B_ymin and B_ymin < A_ymax and B_ymax > A_ymax):  # 8
            #     print("\n\n\n")
            #     print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
            #     print("\n\n\n")
            #     is_accident_happen = True
            #     box = rects[i]

            elif (B_xmin < A_xmin and A_xmin < A_xmax and B_xmax > A_xmax) and (B_ymin < A_ymin and A_ymin < B_ymax and A_ymax > B_ymax): # 9
                print("\n\n\n")
                print(f"{bcolors.FAIL} KAZA BULUNDU 🤕🤕 {bcolors.ENDC}")
                print("\n\n\n")
                is_accident_happen = True
                box = rects[i]    



    return (is_accident_happen, box)


# Hassaslık oranı araç seçimleri için
DEFAULT_CONFIDENCE = 0.5

def draw_rectangles(detections, img):
    rects = []
    for detection in detections:
        _, confidence, coordinates = detection

        if confidence > DEFAULT_CONFIDENCE:
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]

            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            box = (xmin, ymin, xmax, ymax)

            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)

            rects.append(box)
            # print(box)

            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)

    return rects

def draw_circles(objects, img):
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    return img

def draw_errors(img, box):
    xmin, ymin, xmax, ymax = box
    pt1 = (xmin - 5, ymin - 5)
    pt2 = (xmax + 5,ymax + 5)

    cv2.rectangle(img, pt1, pt2, (255, 0, 0), 1)
    cv2.putText(img,
            "KAZA",
            (pt1[0], pt1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            [255, 0, 0],
            2)

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax