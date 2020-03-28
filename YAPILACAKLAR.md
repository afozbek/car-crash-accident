# YAPILACAKLAR

### Object Tracking
- Object tracking algoritması araştırılıcak
- Tespit edilen nesnelere id nasıl atanıyor

Object tracking is the process of:
  - Taking an initial set of object detections (such as an input set of bounding box coordinates)
  - Creating a unique ID for each of the initial detections
  - And then tracking each of the objects as they move around frames in a video, maintaining the assignment of unique IDs

An ideal object tracking algorithm will:

  - Only require the object detection phase once (i.e., when the object is initially detected)
  - Will be extremely fast — much faster than running the actual object detector itself
  - Be able to handle when the tracked object “disappears” or moves outside the boundaries of the video frame
  - Be robust to occlusion
  - Be able to pick up objects it has “lost” in between frames

Algorithms
- centroid tracking with OpenCV
- advanced kernel-based
- correlation-based tracking algorithms.

## centroid tracking algorithm steps
**Euclidean distance.** - Araştırılmalı
**Step #1**: Accept bounding box coordinates and compute centroids


The centroid tracking algorithm assumes that we are passing in a set of bounding box (x, y)-coordinates for each detected object in **every single frame**.

Since these are the first initial set of bounding boxes presented to our algorithm we will assign them unique IDs.

`while video open:`
**Step #2**: Compute Euclidean distance between new bounding boxes and existing objects

**Step #3**: Update (x, y)-coordinates of existing objects

**Step #4**: Register new objects

In the event that there are more input detections than existing objects being tracked, we need to register the new object. “Registering” simply means that we are adding the new object to our list of tracked objects by:
  - Assigning it a new object ID
  - Storing the centroid of the bounding box coordinates for that object

**Step #5**: Deregister old objects

**Ex**:
- https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
- https://medium.com/@manivannan_data/object-tracking-referenced-with-the-previous-frame-using-euclidean-distance-49118730051a

### Velocity Estimitation
- hız tespiti (velocity estimitation) araştırılacak
- Kalman filtresi algoritması araştırılacak, yazı okunacak https://www.ai-articles.net/kalman-filter-for-visual-tracking-cv-project-part-2/

**Ex**:
- https://github.com/Lrakulka/Master_Diploma
- https://www.pyimagesearch.com/2019/12/02/opencv-vehicle-detection-tracking-and-speed-estimation/