import iris_position as ir_pos
import cv2 as cv2
import numpy as np
import settings 
def norm(x):
    return np.sqrt(x.dot(x))

def mean_and_variance(alist):
    an_array = np.array(alist)
   
    mean     = an_array.mean(axis=0)
    var      = an_array.var(axis=0)
    return mean,var

def mean_if_below_error(alist, max_error, debug):
    debug    = True
    an_array = np.array(alist)

    mean     = an_array.mean(axis=0)
    var      = an_array.std(axis=0)
    err      = norm(var)
    if debug:
        print(mean, ' (+/- ', var,  ', magnitude ', err,' )', "shape",mean.shape)

    if( err > max_error):
        return None  
    else:
        return mean 

class iris_position_tracker():
    def __init__(self,camera,nr_samples_per_read=5, err_abs=10, err_rel=10, debug=True):
        self.n             = nr_samples_per_read
        self.max_err_abs   = err_abs
        self.max_err_rel   = err_rel
        self.debug         = debug
        self.camera_stream = cv2.VideoCapture(camera)
        self.frame_buffer  = []
        #self.camera_stream.set(cv2.CAP_PROP_OPENNI_MAX_BUFFER_SIZE , self.n*2)
        
    
    def __iter__(self):
        return self

    def __next__(self):
        self.frame_buffer = []
       
        for i in range(self.n*2):
            frame_available = False
            while not frame_available :
                frame_available , frame   = self.camera_stream.read()
            

            self.frame_buffer.append(frame)

        abs_position_left_buffer  = []
        abs_position_right_buffer = []

        rel_position_left_buffer  = []
        rel_position_right_buffer = []
        
        for frame in self.frame_buffer:
            for i, (abs_l, abs_r, rel_l, rel_r) in enumerate(ir_pos.irides_position_form_video(frame)):
                if i > 1:
                    break
                
                if abs_l is not None and rel_l is not None:
                    abs_position_left_buffer.append(abs_l[:2])
                    rel_position_left_buffer.append(rel_l)

                if abs_r is not None and rel_r is not None:
                    abs_position_right_buffer.append(abs_r[:2])
                    rel_position_right_buffer.append(rel_r)
        
        

        if(len(abs_position_left_buffer)> self.n and len(rel_position_left_buffer)>self.n):

            mean_abs_l = mean_if_below_error(abs_position_left_buffer  , self.max_err_abs, self.debug)
            mean_rel_l = mean_if_below_error(rel_position_left_buffer  , self.max_err_rel, self.debug)
        else:
            mean_abs_l = None
            mean_rel_l = None
            
        if(len(abs_position_right_buffer)> self.n and len(rel_position_right_buffer)>self.n):

            mean_abs_r = mean_if_below_error(abs_position_right_buffer  , self.max_err_abs, self.debug)
            mean_rel_r = mean_if_below_error(rel_position_right_buffer  , self.max_err_rel, self.debug)
        else:
            mean_abs_r = None
            mean_rel_r = None
        
        if self.debug:
            print('res', mean_abs_l, mean_abs_r, mean_rel_l, mean_rel_r)
        return mean_abs_l, mean_abs_r, mean_rel_l, mean_rel_r
       
    def close(self):
        self.camera_stream.release()


if __name__ == "__main__":

    iris_tracker = iris_position_tracker(settings.camera, nr_samples_per_read=3, err_abs=8, err_rel=8, debug=False)
    while True:
        abs_l, abs_r, rel_l, rel_r = next(iris_tracker)

        if(len(iris_tracker.frame_buffer)) > 0:
            frame = iris_tracker.frame_buffer[-1]

            if(abs_l is not None):
                p = tuple(abs_l[:2].astype(np.int))
                r = 1
                cv2.circle(frame, p,r,(0,0,255),2)
                cv2.putText(frame,'left iris',p,cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255)
            
            if(abs_r is not None):
                p = tuple(abs_r[:2].astype(np.int))
                r = 1
                cv2.circle(frame, p,r,(0,0,255),2)
                cv2.putText(frame,'right iris',p,cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255)

            cv2.imshow('out',frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

        

        


