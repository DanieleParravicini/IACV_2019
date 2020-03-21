
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import cv2
import imutils
import time
import dlib
import numpy as np

#surface 3 pro camera 0 
#surface 4 pro camera 1
cap = cv2.VideoCapture(0)

def segment_eyes(frame, eye_left_landmarks, eye_right_landmarks):
    
    eye_left_pts    = np.array(list(map(lambda p: list(p.ravel()), eye_left_landmarks)))
    eye_right_pts   = np.array(list(map(lambda p: list(p.ravel()), eye_right_landmarks)))
    
    max_x_left = min(max(eye_left_pts[:,0]) + 10, frame.shape[1])
    min_x_left = max(min(eye_left_pts[:,0]) - 10, 0)
    max_y_left = min(max(eye_left_pts[:,1]) + 10, frame.shape[0])
    min_y_left = max(min(eye_left_pts[:,1]) - 10, 0)
    #transfor exagon in a square
    eye_left_pts =  np.array([[min_x_left,min_y_left],[max_x_left, min_y_left],[max_x_left,max_y_left],[min_x_left, max_y_left]])

    max_x_right = min(max(eye_right_pts[:,0]) + 10, frame.shape[1])
    min_x_right = max(min(eye_right_pts[:,0]) - 10, 0)
    max_y_right = min(max(eye_right_pts[:,1]) + 10, frame.shape[0])
    min_y_right = max(min(eye_right_pts[:,1]) - 10, 0)

    #transfor exagon in a square
    eye_right_pts =  np.array([[min_x_right,min_y_right],[max_x_right, min_y_right],[max_x_right,max_y_right],[min_x_right, max_y_right]])

    left  = frame[min_y_left:max_y_left,min_x_left:max_x_left]
    right  = frame[min_y_right:max_y_right,min_x_right:max_x_right]
    #print(eye_left_pts)
    
    mask = np.zeros((frame.shape[0], frame.shape[1]))
    
    cv2.fillConvexPoly(mask, eye_left_pts,  1)
    cv2.fillConvexPoly(mask, eye_right_pts, 1)
    
    cv2.imshow('mask', mask)
    
    mask = mask.astype(np.bool)
    out = np.zeros_like(frame)
    out[mask] = frame[mask]  
    cv2.imshow('eyes', out)
    return left,right

def prep_fit_iris(img):
    img     = cv2.equalizeHist(img)
    img     = cv2.medianBlur(img,5)
    return img

def fit_iris(img):
    
    #img = cv2.Canny(img,100,200) 
    cv2.imshow('canny',img)
    rows = img.shape[0]
    circles   = cv2.HoughCircles(np.uint8(img),cv2.HOUGH_GRADIENT,1,rows/8, param1=80, param2=28, minRadius=1, maxRadius=rows)

    if circles is None :
        return 

    '''circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        x = i[0]
        y = i[1]
        r = i[2]
        cv2.circle(img,(x,y),r,(0,255,0),1)
        # draw the center of the circle
        cv2.circle(img,(x,y),2,(0,0,255),2)

    cv2.imshow('detected circles',img)
    print(len(circles[0,:]))'''
    
    return circles[0,0]

def stack(one,two):
    tmp = np.zeros([max(one.shape[0],two.shape[0])*2, max(one.shape[1],two.shape[1])*2, 3], np.uint8)
    tmp[0:one.shape[0], 0:one.shape[1],:] = one
    tmp[0:two.shape[0], one.shape[1]:one.shape[1]+two.shape[1],:] = two
    return tmp

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector(gray)
   
    for i,face in enumerate(faces):
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        landmarks = predictor(frame, face)
        landmarks = face_utils.shape_to_np(landmarks)
        for (x,y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 0,255), -1)
        ##
        cv2.line(frame, tuple(landmarks[36].ravel()), tuple(landmarks[45].ravel()), (0, 255, 0), thickness=3, lineType=8)
        cv2.line(frame, tuple(landmarks[27].ravel()), tuple(landmarks[57].ravel()), (255, 0, 0), thickness=3, lineType=8)
        #fit eyes
        #left 37-42 right 43-48
        left,right = segment_eyes(gray, landmarks[36:42], landmarks[42:48])
        
        #detect iris:
        left   = prep_fit_iris(left)
        c_left = fit_iris(left)
        right  = prep_fit_iris(right)
        c_right = fit_iris(right)
        #plot
        new_left = np.zeros([left.shape[0], left.shape[1],3],np.uint8)
        new_left[:,:,1] = left
        #print(new_left)

        new_right = np.zeros([right.shape[0], right.shape[1],3], np.uint8)
        new_right[:,:,1] = right

        if(c_left is not None):
            cv2.circle(new_left,(c_left[0],c_left[1]),c_left[2],(255,0,0),1)
            cv2.circle(new_left,(c_left[0],c_left[1]),2,(0,0,255),2)
        if(c_right is not None):
            cv2.circle(new_right,(c_right[0],c_right[1]),c_right[2],(255,0,0),1)
            cv2.circle(new_right,(c_right[0],c_right[1]),2,(0,0,255),2)
        cv2.imshow('detected circles_1', new_left)
        cv2.imshow('detected circles',stack(new_left,new_right) )

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
