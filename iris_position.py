from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import cv2
import imutils
import time
import math
import dlib
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import settings
import itertools


from scipy.signal import find_peaks

from head import head_roll, head_scale, project
from Rotator import Rotator
from scipy.signal._peak_finding import peak_prominences


def exagon_padding(pts, padx, pady):
    """
    Function used to add padding to an array
    of points which represent an exagon.
    points are passed via pts
    
    Keyword arguments:
    pts  -- points of the exagon
    padx -- padding x dimension
    pady -- padding in y dimension 
    """
    pts[0,0]-=padx
    pts[1,1]-=pady
    pts[2,1]-=pady
    pts[3,0]+=padx
    pts[4,1]+=pady
    pts[5,1]+=pady
    return pts

def bounding_box(pts,  width, height, padx=0, pady=0):
    """
    Extract the left,right,top,bottom 
    coordinates in a list of points.
    It can apply padding

    Width 
    Height
    padx -- padding in x dimension
    pady -- padding in y dimension

    """
    left  = max(min(pts[:,0]) - padx, 0)
    right = min(max(pts[:,0]) + padx, width)
    top   = max(min(pts[:,1]) - pady,0)
    bottom= min(max(pts[:,1]) + pady, height)
   
    return left,right,top,bottom

def extract_polyline(img, poly_pts):
    """
    Function that extracts part of image 
    contained in the convex hull of the points 
    given as second arguments

    it returns:
    - part of the image contained innside the polygon described by poly_pts
    - a mask which signal what pixels are considered valid
    """
    #extract left,right,top,bottom coordinate
    l,r,t,b   = bounding_box(poly_pts, img.shape[1], img.shape[0])
    #extract the image part which fall inside the bounding box
    sub_img   = np.array(img[t:b+1,l:r+1])
    #convert the original points in coordinate
    # w.r.t. top,left coordinate 
    poly_pts  = np.array(list(map(lambda p: [p[0]-l,p[1]-t], poly_pts)))
    # the process can be simplified by considering  a mask matrix 
    #
    # initialize a mask to 0
    mask      = np.zeros_like(sub_img)
    # set to 1 mask bits inside the pts supplied as arguments
    cv2.fillConvexPoly(mask, poly_pts,  1)
    # consider mask bits as boolean
    mask = mask.astype(np.bool)
    #define a matrix to hold result
    out       = np.zeros_like(sub_img)
    #fill the result matrix
    out[mask] = sub_img[mask]
    
    return out, mask

def segment_eye(frame, eye_landmarks, square):
    """
    Function that returns image related to eye

    Returns:
    1. image segmented
    2. mask which pixel has to be considered
    3. top,left coordinates w.r.t. original image
    """
    padx, pady = (3,3)
    eye_pts    = np.array(list(map(lambda p: list(p.ravel()), eye_landmarks)))


    if square :
        #extract bounding box, with padding if required. 
        l,r,t,b = bounding_box(eye_pts, frame.shape[1], frame.shape[0],padx,pady)
        #extract a square submatrix
        eye     = np.array(frame[t:b,l:r])
        #define an easy mask all 1.
        mask    = np.ones_like(eye)
        mask    = mask.astype(np.bool)

    else:
        #pads an exgon shaped point array
        eye_pts     = exagon_padding(eye_pts, padx,pady)
        #obtain  bounding box coordinates with special function due to the fact that is an exagon
        l,_,t,_     = bounding_box(eye_pts, frame.shape[1], frame.shape[0],padx,pady)
        #extract subimage with special  function due to the fact that is an exagon
        eye, mask   = extract_polyline(frame, eye_pts)

    return eye, mask, (t,l)

def get_left_eye_landmarks(face_landmarks):
    return face_landmarks[36:42]

def get_right_eye_landmarks(face_landmarks):
    return face_landmarks[42:48]

def segment_eyes(frame,face_landmarks, square=True):
    """
    Function that returns the image part related to eyes. 

    Parameteres:
    -  square : choose whether image has to return a valid image 
       subset which has to be either a square or an exagon
    """
    #fit eyes
    #landmarks for the eyes are
    # - left 37-42
    # - right 43-48
    eye_left_landmarks  =  get_left_eye_landmarks(face_landmarks)
    eye_right_landmarks =  get_right_eye_landmarks(face_landmarks)

    return segment_eye(frame, eye_left_landmarks, square) , segment_eye(frame, eye_right_landmarks, square)

def prep_fit_iris(img, rscale_factor=5):
    erode_dilate = True
    img          = cv2.equalizeHist(img)
    img          = rescale(img, rscale_factor)

    if(erode_dilate):
        #eliminate artifacts such as reflections
        # Creating kernel 
        kernel_height   = 3 #max(img.shape[0]//8,5)
        kernel          = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_height,kernel_height))
        img             = cv2.dilate(img,kernel)
        img             = cv2.erode(img, kernel) 
    
    #cv2.imshow('test',rescale(img,8))


    if(rscale_factor > 1):
    
        a       = img.shape[0] //16
        a       = max(5,a)
        a       = a+1 if a % 2 == 0 else a

        #img     = cv2.GaussianBlur(img,(a,a),1)
        #img     = cv2.medianBlur(img,a,1)

    return img

def fit_iris_with_HoughT(img, mask):
    """
    Extract Iris position using hough transform methods
    Parameters:
    Returns:
    x - eye centre position on x axis
    y - eye centre position on y axis
    r - eye radius estimation
    """
    debug = False

    height = img.shape[0]
    circles   = cv2.HoughCircles(np.uint8(img),cv2.HOUGH_GRADIENT,1,height//10, param1=40, param2=24, minRadius=height//8, maxRadius=height//2)
    
    if circles is None :
        return
    
    if debug:
        new_tmp = np.zeros([img.shape[0], img.shape[1],3], np.uint8)
        new_tmp[:,:,1] = img

    circles = np.uint16(np.around(circles))
    m = {}
    for i,c in enumerate(circles[0,:]):
        # draw the outer circle
        x = c[0]
        y = c[1]
        r = c[2]
        # compute mean value in that circle
        v = get_mean_value_inside_circle(img,x,y,r)
        # avoid considering circles that have average colour intensity 
        # that are over a certain thresold.
        if v > 70:
            continue
        # neglect circles that are centered 
        # in a non valid pixel
        if not mask[y,x] :
            continue
        #Add on dictionary 
        m[i] = v
        
        if debug:
            cv2.circle(new_tmp,(x,y),r,(0,255,0),1)
            #draw the center of the circle
            cv2.circle(new_tmp,(x,y),2,(0,0,255),2)
            #and write average pixel value on the image
            cv2.putText(new_tmp,str(i)+':'+str(v),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255)

    #sort depending on the average value 
    t = sorted(m.items(), key=lambda x: x[1])
    #if no circle is detected return None
    if len(t) == 0:
        return None
    
    if debug:
        cv2.imshow('detected circles debug',new_tmp)
   
    #return [x,y,r] of the most probable iris point
    return circles[0,t[0][0]]

def get_mean_value_inside_circle(img,xc,yc,radius):
    
    y,x = np.ogrid[0:img.shape[0], 0:img.shape[1]]
    mask = ( ( (x-xc)**2 + (y-yc)**2 ) <= radius**2) 
    mean = img[mask].mean()

    return mean

def IPF_x(img,mask,x):
    #this sum along X axis is not affected by the fact that some 
    #elements are invalid. They would be zero, and the division
    #would change accordingly.
    return img[:,x].sum() /mask[:,x].sum()

def VPF_x(img, mask,x, IPF):
    #this sum along X axis is affected by the fact that some 
    #elements are invalid. 
    #To handle them multiply corresponding values with their mask 
    # value, if they're valid a 1*v would leave the value unchanged 
    # if they're invalid a 0*v would ignore that value.
    delta = (np.square(img[:,x] - IPF[x]))
    #delta = (np.abs(img[:,x] - IPF[x]))
    delta = delta*np.transpose(mask[:,x]) 

    return (delta).sum() / mask[:,x].sum() 

def IPF_y(img,mask,y):
    #this sum along Y axis is not affected by the fact that some 
    #elements are invalid. They would be zero, and the division
    #would change accordingly.
    return img[y,:].sum() /mask[y,:].sum()

def VPF_y(img, mask, y, IPF):
    #this sum along Y axis is affected by the fact that some 
    #elements are invalid. 
    #To handle them multiply corresponding values with their mask 
    # value, if they're valid a 1*v would leave the value unchanged 
    # if they're invalid a 0*v would ignore that value.
    delta = (np.square(img[y,:] - IPF[y]))
    #delta = (np.abs(img[y,:] - IPF[y]))
    delta = delta*np.transpose(mask[y,:]) 
    return (delta).sum() / mask[y,:].sum() 

def HPF_boundaries(img,GPF_values,IPF_values, debug=False):
    debug          = False
    use_derivative = True

    approximate_eye_width   = img.shape[0]
    spacing = 3
    derivate_GPF_values     = np.abs(np.gradient(GPF_values, spacing))
   
    vertical = (img.shape[0] == len(GPF_values))
    min_eye_width           = max(approximate_eye_width *(0.6 if vertical else 0.8) ,1)
    max_eye_width           = max(approximate_eye_width*1.2,1)
    #print('between ', min_eye_width, ' and ', max_eye_width)
    #if min_eye_width < 3:
    #    return None,None
    
    minimum_peak_prominence = 0.6
    
    if debug:
        #this is done in order to not force square pixels in the image
        gs_kw  = dict(width_ratios=[1], height_ratios=[1,1, 1,img.shape[1]/img.shape[0] if vertical else 1 ])
        _, axs = plt.subplots(ncols=1, nrows=4, constrained_layout=True,  gridspec_kw=gs_kw)
        axs[0].plot(GPF_values,'r')
        axs[1].plot(derivate_GPF_values, 'r')
        axs[2].plot(IPF_values,'g')

    #first detect topmost n peaks
    peaks, peak_properties  = find_peaks( derivate_GPF_values if use_derivative else GPF_values,prominence=minimum_peak_prominence, distance=approximate_eye_width//4)
    #plot detected peaks
    if(debug):
        for p in peaks:
            for ax in axs[:-1]:
                ax.axvline(p, color='r')
    n_top_peaks             = 4
    

    if len(peaks) >= n_top_peaks :
        #take n_top_most peaks with greatest "prominence", i.e. greatest elevation w.r.t. their neighbourhood
        prominences = peak_properties["prominences"]
       
        #to extract the elements
        peak_indices         = np.argpartition(-1*prominences, n_top_peaks-1)[:n_top_peaks]
        peaks_location       = [peaks[i] for i in peak_indices] 
        peaks                = sorted(peaks_location)

    pad = 1
    boundaries_positions = list(itertools.chain([pad],peaks,[len(GPF_values)-pad]))

    #plot selected boundaries
    if(debug):
        for p in boundaries_positions:
            for ax in axs[:-1]:
                ax.axvline(p, color='g')

    possible_boundaries = [(a, b) for a in boundaries_positions for b in boundaries_positions] 
    possible_boundaries = list(filter( lambda t: (t[1] - t[0]) >= min_eye_width and (t[1] - t[0]) <= max_eye_width , possible_boundaries))
    
    if len(possible_boundaries) == 0 :
        if debug:
            print('nothing with distance between ', min_eye_width, ' and ', max_eye_width)
        first,second = None, None
    else:
        
        possible_boundaries_mean_value = list(map( lambda t :  sum(IPF_values[t[0]:t[1]])/(1+t[1]- t[0]) ,possible_boundaries))
       
        tmp = list(zip(possible_boundaries, possible_boundaries_mean_value))
        tmp = filter( lambda x: x[1] < 160, tmp)
        tmp = list(tmp)
        
        if len(tmp) == 0 :
            first,second = None, None
        else:
            possible_boundaries, possible_boundaries_mean_value = list(zip(*tmp))

            region_with_smallest_mean_value = np.argmin(possible_boundaries_mean_value)
            first,second = possible_boundaries[region_with_smallest_mean_value]

        

    #print debug information
    if debug:
        if first is not None:
            for ax in axs:
                ax.axvline(first, color='b')
            
        if second is not None:
            for ax in axs:
                ax.axvline(second, color='b')
       
        #if the image is vertical rotate the image
        if vertical:
            axs[-1].imshow(np.transpose(np.array(img)), aspect='auto')
        else :
            axs[-1].imshow(img, aspect='auto')

        plt.show()

    return first,second

def fit_iris_with_HPF(img, mask, debug=False):
    """
    Extract Iris position using hybrid projective function
    Parameters:

    Returns:
    x - eye centre position on x axis
    y - eye centre position on y axis
    r - eye radius estimation
    """
    
    #select which method to use: 
    # True: selects the two points with highest positive and negative derivative
    # False: selects the two points based on threshold on (value_max-value_min)*th_factor+value_min
    
    alpha          = 0.6

    if debug:
        #print(img.shape, mask.shape)
        cv2.imshow('mask',mask*255)
        cv2.imshow('eye in input',img)
    
    #Y axis
    #compute IPF, VPF values 
    IPF_values = list(map(lambda y  : IPF_y(img,mask,y)             , range(img.shape[0])))
    VPF_values = list(map(lambda y  : VPF_y(img,mask,y, IPF_values) , range(img.shape[0])))
    
    #combine IPF and VPF values with alpha coefficient prescribed by paper
    GPF_values = np.array(VPF_values) *np.float(alpha) - np.array(IPF_values) * np.float(1 - alpha) 

    first_y, second_y = HPF_boundaries(img,GPF_values, IPF_values)
    if first_y is None or second_y is None:
        return None

    ############
    ############
    ############


    #X axis
    IPF_values = list(map(lambda x  : IPF_x(img,mask,x)             , range(img.shape[1])))
    VPF_values = list(map(lambda x  : VPF_x(img,mask,x, IPF_values) , range(img.shape[1])))
    GPF_values = np.float(alpha) * np.array(VPF_values) - np.float(1 - alpha) * np.array(IPF_values)

    first_x,second_x = HPF_boundaries(img,GPF_values,IPF_values)
    if first_x is None or second_x is None:
        return None
    #computed expected x,y,r of the iris
    x = (first_x+second_x)//2
    r = np.abs(second_x-first_x)//2
    y = (first_y+second_y)//2

    
    if debug :
        #represent results in an image.
        # using lines
        cv2.line(img, (first_x,0 ),(first_x,img.shape[0]) , 255)
        cv2.line(img, (second_x,0),(second_x,img.shape[0]), 255)
        cv2.line(img, (0,first_y) ,(img.shape[1],first_y) ,255)
        cv2.line(img, (0,second_y),(img.shape[1],second_y),255)
        # using a circle
        cv2.circle(img, (x,y),r,(255,0,0),1)

        
        cv2.imshow('eye',rescale(img,4) )
 
    #return [x,y,r] of the most probable iris point
    return [x,y,r]

def rescale(img, scale_percent):
    
    width = int(img.shape[1] * scale_percent )
    height = int(img.shape[0] * scale_percent)
    
    dimR = (width, height)

    # resize image
    return cv2.resize(img, dimR, cv2.INTER_LINEAR) #.INTER_NEAREST)  #cv2.INTER_CUBIC )#interpolation=cv2.INTER_AREA )

def irides_position(frame, face_landmarks):
    """
    Receives a frame in grayscale and some face landmarks of a person in the image
    and extract iris positions


    """	
    debug               = False
    use_HPF 			= True
    square              = True

    roll_angle      	= head_roll(face_landmarks)

    image_center    	= tuple(np.array(frame.shape[1::-1]) / 2)
    rot             	= Rotator(image_center,roll_angle)
    rotated_frame   	= rot.transform(frame)
    rotated_landmarks 	= rot.transform_point(face_landmarks)
    
    c_left, c_right = irides_position_relative_to_rotated_img(rotated_frame=rotated_frame, rotated_landmarks=rotated_landmarks, square=square, use_HPF=use_HPF,debug=debug)

    #Note we have to recall irides_position_relative_to_eye_extreme here
    #because c_left,c_right have to be expressed w.r.t. to the entire image
    # and to keep simple the computations we operate with a rotated image
    iris_rel_left, iris_rel_right = irides_position_relative_to_eye_extreme(rotated_landmarks, (c_left,c_right))


    #Since we have rotated the image we have to rotate back the point
    if(c_left is not None):
        c_left[:2]   = rot.reverse_transform_point(np.array([c_left[:2]]))

    if(c_right is not None):
        c_right[:2]  = rot.reverse_transform_point(np.array([c_right[:2]]))		

    return c_left, c_right , iris_rel_left, iris_rel_right

def irides_position_relative_to_rotated_img(rotated_frame, rotated_landmarks, square, use_HPF, debug):
    debug = True
    (left, mask_left, (top_left,left_left)),(right, mask_right, (top_right, left_right)) = segment_eyes(rotated_frame, rotated_landmarks, square)

    is_left_closed, is_right_closed = are_eyes_closed(rotated_landmarks)
    # detect iris based on Hough Transform fail to recognize iris with 
    # small images require a resize but HPF performs better without resize
    
    factor_magnification = 5 if not use_HPF else 1

    if is_left_closed:
        c_left = None
    else:
        left   			= prep_fit_iris(left,factor_magnification)
        mask_left    	= rescale(mask_left.astype(np.uint8),factor_magnification)
       
        if (use_HPF):
            c_left = fit_iris_with_HPF(left, mask_left)
        else:
            c_left = fit_iris_with_HoughT(left,mask_left)

    if is_right_closed: 
        c_right = None
    else:
        right         = prep_fit_iris(right,factor_magnification)
        mask_right    = rescale(mask_right.astype(np.uint8),factor_magnification)
        if (use_HPF):
            c_right = fit_iris_with_HPF(right, mask_right)
        else:
            c_right = fit_iris_with_HoughT(right,mask_right)

    

    #At this point c_left and c_right are x,y coordinate referring 
    #to the subimage reference system
    #Moreover they have to be rescaled according yo factor_magnification
    if(c_left is not None):
        if debug :
            cv2.circle(left,tuple(c_left[:2]),c_left[2],(255,0,0),1)

        c_left[0] = c_left[0]/factor_magnification  + left_left
        c_left[1] = c_left[1]/factor_magnification  + top_left
        c_left[2] = c_left[2]/factor_magnification

        c_left = np.array(c_left)
     #here the bugfix but gives problems
    if(c_right is not None):
        if debug:
            cv2.circle(right,tuple(c_right[:2]),c_right[2],(255, 0, 0),1)

        c_right[0] = c_right[0]/factor_magnification + left_right
        c_right[1] = c_right[1]/factor_magnification + top_right
        c_right[2] = c_right[2]/factor_magnification

        c_right = np.array(c_right)

    
    if debug and 0 not in left.shape and 0 not in right.shape :
        left  = cv2.resize(left,  (400,200))
        right = cv2.resize(right, (400,200)) 
        frame_illustration =  stack(left,right)
        cv2.imshow('position',frame_illustration )
    
    return c_left, c_right

def stack(one,two):
    tmp = np.zeros([max(one.shape[0],two.shape[0]), max(one.shape[1],two.shape[1])*2], np.uint8)
    tmp[0:one.shape[0], 0:one.shape[1]] = one
    tmp[0:two.shape[0], one.shape[1]:one.shape[1]+two.shape[1]] = two
    return tmp

def are_eyes_closed(rotated_landmarks):
    left_eye_landmarks  = get_left_eye_landmarks(rotated_landmarks)
    right_eye_landmarks = get_right_eye_landmarks(rotated_landmarks)

    return is_eye_closed(left_eye_landmarks, head_scale(rotated_landmarks)), is_eye_closed(right_eye_landmarks, head_scale(rotated_landmarks))

def is_eye_closed(eye_landmarks, scale):
    
    """
    Given eye landmarks and a scale factor of the face
    tells if the eye is closed or not.

    Keyword arguments:
    eye_landmarks -- landmark of the eye
    scale -- relative distance of the head from the screen (1 head as close as possible to the screen)
    """

    eye_top_landmark           = eye_landmarks[2]
    eye_bottom_landmark        = eye_landmarks[4]

    H = np.abs((eye_bottom_landmark[1]-eye_top_landmark[1]))/scale

    if(H<=13):
        return True

def irides_position_relative_to_eye_extreme(face_landmarks, irides_position):

    left_eye_landmarks  = get_left_eye_landmarks(face_landmarks)
    right_eye_landmarks = get_right_eye_landmarks(face_landmarks)

    left_relative_position  = iris_position_relative_to_eye_extreme(left_eye_landmarks, irides_position[0]) 
    right_relative_position = iris_position_relative_to_eye_extreme(right_eye_landmarks,irides_position[1]) 

    return left_relative_position, right_relative_position

def iris_position_relative_to_eye_extreme(eye_landmarks, iris_position):

    """
    Given the eye landmarks and the iris position returns
    a local position relative to the left and top edges of the eye.
    Example:
     x_l = 0.6    x_r = 0.6
    (-----x---)  (-----x---)
    Same for the y axis.

    Keyword arguments:
    eye_landmarks -- landmark or the eye
    iris_position -- absolute iris position relative to the image frame

    """
    if(iris_position is None):
        return None

    eye_external_landmark   = eye_landmarks[0]
    eye_internal_landmark   = eye_landmarks[3]
    eye_centre              = (eye_external_landmark+eye_internal_landmark)/2

    iris_position = np.array(iris_position[:2])

    ratio_posit   = (iris_position[:2] - eye_centre) 
    
    return ratio_posit

def irides_position_form_video(frame):
    """
    Given a camera_number (0 for surface pro 3 and 1 for surface pro 4)
    The function will return the eye position in the current image frame wrt
    the nose side eye angle.
    """
   
    frame_equalized = clahe(frame)
    gray            = cv2.cvtColor(frame_equalized, cv2.COLOR_BGR2GRAY)
    faces           = detector(gray)

    for i, face in enumerate(faces):
        landmarks = predictor(frame_equalized, face)
        landmarks = face_utils.shape_to_np(landmarks)
        yield  irides_position(gray, landmarks)


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


#this part will be executed anyhow
#is a intialization
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#predictor = dlib.shape_predictor("new_shape_predictor_68_face_landmarks.dat")

if __name__ == "__main__":
    #this code will be executed only if we do not load this file as library

    #surface 3 pro camera 0
    #surface 4 pro camera 1
    cap = cv2.VideoCapture(settings.camera)

    debug = False

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out_width  = 1080
    out_height = 720
    out = cv2.VideoWriter("output.avi", fourcc, 10.0, (out_width, out_height))


    while True:
        _, frame = cap.read()
        
        frame_equalized = clahe(frame)
        gray    = cv2.cvtColor(frame_equalized, cv2.COLOR_BGR2GRAY)
        faces   = detector(gray)

        for i,face in enumerate(faces):
            landmarks = predictor(frame_equalized, face)
            landmarks = face_utils.shape_to_np(landmarks)
            
            iris_left,iris_right,iris_rel_left,iris_rel_right   = irides_position(gray, landmarks )
            
            #note: we can safely modify frame as it is not passed to iris position.
            if(iris_left is not None):
                p = tuple(iris_left[:2].astype(np.int))
                r = int(iris_left[2])
                cv2.circle(frame, p,r,(0,0,255),2)
                cv2.putText(frame,str(i)+' : left iris',p,cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255)
            
            if(iris_right is not None):
                p = tuple(iris_right[:2].astype(np.int))
                r = int(iris_right[2])
                cv2.circle(frame, p,r,(0,0,255),2)
                cv2.putText(frame,str(i)+' : right iris',p,cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255)

            print(i, " : (",iris_rel_left, iris_rel_right, ")" )
            if debug:
                for (x,y) in landmarks:
                    cv2.circle(frame, (x,y), 1, (0, 0,255), -1)


        key = cv2.waitKey(1)
        if key == 27:
            break

        cv2.imshow('out',frame)
        frame_out = cv2.resize(frame, (out_width, out_height))
        out.write(frame_out)

    cap.release()
    cv2.destroyAllWindows()
    out.release()
