from filterpy.kalman import KalmanFilter
import numpy as np

class kalmanFilter():
    def __init__(self, state_dim, measurement_dim, dt, Q, R):
 
        
        self.kalman = KalmanFilter(dim_x=state_dim, dim_z=measurement_dim)

        #state variables
        self.kalman.x = np.zeros((state_dim,1))
        
        self.kalman.F = np.eye(state_dim,state_dim)
        
        L = int(state_dim/2)
        
        temp = np.eye(L,L)*dt
        
        self.kalman.F[0:L,L:] = temp
        self.kalman.H = np.zeros((measurement_dim,state_dim))
        self.kalman.H[0:measurement_dim,0:measurement_dim] = np.eye(measurement_dim,measurement_dim)

        self.kalman.P = np.eye(state_dim,state_dim)*100
        
        self.kalman.R = np.eye(measurement_dim,measurement_dim)*R
        
        self.kalman.Q = np.eye(state_dim,state_dim)*Q
        self.kalman.Q[0:L,0:L] = np.zeros((L,L))


        
    def predict(self, z):
        
        #z: measurements
        
        self.kalman.predict()
        self.kalman.update(z)
        return self.kalman.x.T



class localization():
    def __init__(self, frame_rate, width, height):
        
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self.kalman_speed = kalmanFilter(state_dim=6, measurement_dim=3,
                                         dt=1/self.frame_rate, Q=0.001, R=50)
        self.reduction_rate = 0.8
        self.fov_h = 100
        self.fov_v = 70
    
    def predict(self, boundingBox, real_length):

        (x,y,w,h) = boundingBox   
        #print('loc: {:.2f}, {:.2f}'.format(loc[0], loc[1]))
        depth = self.calculate_dist(w*self.reduction_rate, real_length)
        loc = (x+w/2,y+h/2)
        (px,py,pz) = self.calx_xyz(depth, loc)
        ret = self.kalman_speed.predict(np.array([px,py,pz]))
        (kx,ky,kz,kvx,kvy,kvz) = ret.ravel()
        pos = (kx,ky,kz)
        vel = (kvx,kvy,kvz)
        return pos, vel
    
    def calculate_dist(self, pixels, real_length):
        alpha = 329
        beta = 2
        gamma = 0.4

        return (-1/beta*np.log(pixels/alpha))**(1/gamma) * real_length/0.082;
    
    def calx_xyz(self, depth, search_point):
        (i,j) = search_point
        fh = self.fov_h/180*np.pi
        fv = self.fov_v/180*np.pi
        
        px = depth * (i/(self.width-1)-0.5)*np.tan(fh/2) 
        py = depth * (0.5-j/(self.height-1))*np.tan(fv/2)
        pz = depth
        return (px,py,pz)

import cv2
import numpy as np
from kalmanFilter import kalmanFilter

class customTracker():
    def __init__(self):
        self.kernel = np.ones((3,3))
        self.width = 0
        self.height = 0
        self.tracking_method = cv2.TM_CCOEFF_NORMED
        self.boundingBox = [] 
        self.acq_offset = 20
        self.acq_scale = 1.5
        self.match_threshold = 0.5
        self.refresh_threshold = 0.5
        self.frame_rate = 30
        self.small_image_scale = 0.9
        self.large_image_scale = 1.1
        self.acq_pen_color = (255,255,255)
        self.acq_pen_thickness = 2
        self.selected_img = []
        self.found_x = 0
        self.found_y = 0
        self.search_point = (0,0)
        self.track_loss = False        
        self.kalman_tracker = kalmanFilter(state_dim=4, measurement_dim=2,
                                         dt=1, Q=10, R=1)        
        self.lost_box = (0, 0, 20, 20)
    
    def clear(self):
        self.__init__()

    def init(self, frame, boundingBox):
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = cv2.equalizeHist(frame)
        (self.height, self.width) = frame.shape
        self.boundingBox = boundingBox 
        (x,y,w,h) = self.boundingBox
        self.selected_img = frame[y:y+h,x:x+w]
        self.found_x = x
        self.found_y = y
        self.search_point = (self.found_x+w/2, self.found_y + h/2)
        
        
    def update(self, frame):
        frame_new = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   
        frame_new = cv2.equalizeHist(frame_new)
        if self.track_loss:
            return False, self.lost_box
        ret = True
        max_val, top_left = self.inner_temp_match(frame_new,self.selected_img, self.boundingBox)
        
        if max_val < self.match_threshold:
           
            rescaled_img_larger, larger_bounding_box = self.adapt_img_size(self.large_image_scale,0)
        
            max_val_larger, top_left_larger = self.inner_temp_match(frame_new, rescaled_img_larger, larger_bounding_box)
            
            rescaled_img_smaller, smaller_bounding_box = self.adapt_img_size(self.small_image_scale,1)
            
            max_val_smaller, top_left_smaller = self.inner_temp_match(frame_new, rescaled_img_smaller, smaller_bounding_box)

            
            print('min max: ',max_val_larger, max_val_smaller)
            max_val = max(max_val_larger, max_val_smaller)
            
            if max_val > self.match_threshold:
                if max_val_larger > max_val_smaller:
                    top_left = top_left_larger
                    self.selected_img = rescaled_img_larger
                    self.boundingBox = larger_bounding_box
                    if max_val > self.refresh_threshold:
                        self.refresh_selected_img(frame_new,0)
                  
                else:
                    top_left = top_left_smaller
                    self.selected_img = rescaled_img_smaller
                    self.boundingBox = smaller_bounding_box 
                    if max_val > self.refresh_threshold:
                        self.refresh_selected_img(frame_new,1)

            else:
                ret = False
                self.track_loss = True
                return False, self.lost_box

            
        print('corr: {:.2f}'.format(max_val))
        
        (xx,yy,ww,hh) = self.get_acq_box(self.boundingBox)
        
        (_,_,w,h) = self.boundingBox
        
        kalman_res = self.kalman_tracker.predict(np.array(top_left))
        
        (inner_pos_x, inner_pos_y, vx, vy) = kalman_res.ravel()
        
        self.found_x = inner_pos_x + xx
        self.found_y = inner_pos_y + yy
        self.search_point = (self.found_x + w/2, self.found_y + h/2 )
        return ret, (self.found_x, self.found_y, w, h)


    def plot_acq_box(self,frame):
        (xx,yy,ww,hh) = self.get_acq_box(self.boundingBox)
        cv2.rectangle(frame, (xx, yy), (xx + ww, yy + hh), self.acq_pen_color, self.acq_pen_thickness)
        cv2.putText(frame,'Acquisition Gate', org=(xx-5,yy-5),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=0.5,color=self.acq_pen_color)

        
    def get_acq_box(self, bounding_box):
        (x,y,w,h) = bounding_box
        ww = int(w*self.acq_scale + self.acq_offset)
        hh = int(h*self.acq_scale + self.acq_offset)
        
        xx = int(self.search_point[0] - ww/2)
        yy = int(self.search_point[1] - hh/2)
        
        xx = max(0,xx)
        yy = max(0,yy)
        
        if xx + ww >= self.width:
            xx = self.width - ww -1
            
        if yy + hh >=self.height:
            yy = self.height - hh -1
        
        return (xx,yy,ww,hh)
    
    def adapt_img_size(self, img_scale, scale_type):
        (x,y,w,h) = self.boundingBox        
        w_new = int(w*img_scale)
        h_new = int(h*img_scale)
        
        if scale_type == 0:
            residue_x = - (img_scale-1) / 2 * w_new
            residue_y = - (img_scale-1) / 2 * h_new * self.height/self.width
        else:
            residue_x = (1-img_scale) / 2 * w_new
            residue_y = (1-img_scale) / 2 * h_new * self.height/self.width
            
        x = int(self.found_x + residue_x)
        y = int(self.found_y + residue_y)

        new_bounding_box = (x,y,w_new,h_new)              
        rescaled_img = cv2.resize(self.selected_img,(w_new,h_new))
        return rescaled_img, new_bounding_box
    
    
    def inner_temp_match(self, frame, template_img, bounding_box):
        
        (xx,yy,ww,hh) = self.get_acq_box(bounding_box)
        frame_new = frame[yy:yy+hh, xx:xx+ww]
        (x,y,w,h) = bounding_box
        res = cv2.matchTemplate(frame_new, template_img, self.tracking_method)
        min_val,max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        return max_val, top_left

    def refresh_selected_img(self, frame, scale_type):
        (x,y,w,h) = self.boundingBox
    
        
        self.selected_img = frame[y:y+h,x:x+w]

        
lass customMultiTracker():
    
    def __init__(self):
        self.tracker_list = []
    
    def add(self, tracker):
        self.tracker_list.append(tracker)
        
    def clear(self):
        self.tracker_list = []
        
    def update(self,frame):
        box_list = []

        res = True
        for tracker in self.tracker_list:
            if tracker.track_loss == False:
                ret, box = tracker.update(frame)
                if ret:
                    res = True
                    box_list.append(box)
        
        return res, box_list
