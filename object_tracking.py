
from ctypes import *
import math
import random

import numpy as np
from numpy.linalg import norm
import cv2

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



colors = [tuple(255 * np.random.rand(3)) for _ in range(15)]
  

lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.25, hier_thresh=.5, nms=.45):
    im = load_image(image.encode('utf-8'), 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res


detected_objects = ['car']
font = cv2.FONT_HERSHEY_SIMPLEX
net = load_net("./cfg/yolov3.cfg".encode('utf-8'), "./yolov3.weights".encode('utf-8'), 0)
meta = load_meta("./cfg/coco.data".encode('utf-8'))
videoPath = "./data/videos/1.mp4"
less = 100
cap = cv2.VideoCapture(videoPath)
def main():
    ref_frame_axies = []
    ref_frame_label = []
    label_cnt = 1
    min_distance = 50
    while(True):
        ret,img = cap.read()
        if ret == True:
            cur_frame_axies = []
            cur_frame_label = []
            cv2.imwrite('test.jpg',img)
            outputs = detect(net, meta, "test.jpg")
            for color,output in zip(colors,outputs):
                text = output[0].decode('utf-8')
                x = int(output[2][0])
                y = int(output[2][1])
                fw = int(output[2][2])
                fh = int(output[2][3])
                w = int(fw/2)
                h = int(fh/2)
                acc = int(output[1] * 100)
                left = y - h
                top = x - w
                right = y + h
                bottom = x + w
                lbl = float('nan')
                if text in detected_objects:
                    if(len(ref_frame_label) > 0):
                        b = np.array([(x,y)])
                        a = np.array(ref_frame_axies)
                        distance = norm(a-b,axis=1)
                        min_value = distance.min()
                        if(min_value < min_distance):
                            idx = np.where(distance==min_value)[0][0]
                            lbl = ref_frame_label[idx]
                    if(math.isnan(lbl)):
                        lbl = label_cnt
                        label_cnt += 1
                    cur_frame_label.append(lbl)
                    cur_frame_axies.append((x,y))
                    cv2.rectangle(img,(top,left),(bottom,right),color,2)
                    cv2.putText(img,'{}{}-{}%'.format(text,lbl,acc),(top,left), font, 1,(255,255,255),2)
            ref_frame_label = cur_frame_label
            ref_frame_axies = cur_frame_axies
            cv2.imshow('image',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()