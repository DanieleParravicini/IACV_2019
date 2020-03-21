
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
    
    #print(eye_left_pts)
    
    mask = np.zeros((frame.shape[0], frame.shape[1]))
    
    cv2.fillConvexPoly(mask, eye_left_pts,  1)
    cv2.fillConvexPoly(mask, eye_right_pts, 1)
    
    cv2.imshow('mask', mask)
    
    mask = mask.astype(np.bool)
    out = np.zeroes_like(frame)
    out[mask] = frame[mask]  
    
    return out

def fit_iris(img):

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,4)
    if circles is None :
        return 

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('detected circles',cimg)

    return 


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
        out = segment_eyes(gray, landmarks[36:42], landmarks[42:48])
        cv2.imshow('eye_'+str(i), out)
        #detect iris:
        fit_iris(out)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
