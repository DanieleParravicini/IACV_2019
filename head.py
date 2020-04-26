
import math


import cv2
import imutils
import numpy as np

from imutils import face_utils
from imutils.video import VideoStream

import dlib


def head_roll(face_landmarks):                      #rotation along the axis perpendicular to the plane
    D = math.sqrt((face_landmarks[36][0]-face_landmarks[45][0])** 2+(face_landmarks[36][1]-face_landmarks[45][1])**2)
    return math.asin((face_landmarks[45][1]-face_landmarks[36][1])/D)*180/math.pi

def head_pitch(face_landmarks):                     # rotation along the orizontal axis TO BE CALIBRATED
    extra_point_x = face_landmarks[21][0]+(face_landmarks[22][0]-face_landmarks[21][0])/2
    extra_point_y = face_landmarks[21][1]+(face_landmarks[22][1]-face_landmarks[21][1])/2

    D = math.sqrt((extra_point_x-face_landmarks[57][0])** 2+(extra_point_y - face_landmarks[57][1])**2)
    centre_point = line_intersection((face_landmarks[36], face_landmarks[45]), ((extra_point_x, extra_point_y), face_landmarks[57]))
    bottom_to_centre = math.sqrt(((face_landmarks[57][0]-centre_point[0]) ** 2)+(face_landmarks[57][1]-centre_point[1])**2)
    return 47-math.asin(bottom_to_centre/D)*180/math.pi

def head_yaw(face_landmarks):                       #rotation along the vertical axis TO BE CALIBRATED
    D = math.sqrt((face_landmarks[36][0]-face_landmarks[45][0])** 2+(face_landmarks[36][1]-face_landmarks[45][1])**2)
    centre_point = line_intersection((face_landmarks[36], face_landmarks[45]), (face_landmarks[27], face_landmarks[57]))
    left_to_centre = math.sqrt(((face_landmarks[36][0]-centre_point[0]) ** 2)+(face_landmarks[36][1]-centre_point[1])**2)
    return math.asin(left_to_centre/D)*180/math.pi-30

def project(line_point_1, line_point_2, point, D, H):
    a = point-line_point_1
    A = math.sqrt((line_point_1[0]-point[0])**2 +(line_point_1[1]-point[1])**2)
    d = line_point_2-line_point_1
    cos_alpha = np.dot(a, d)/(D*A)
    sin_alpha = math.sqrt(1-cos_alpha**2)
    return (A * cos_alpha, H/2 - A * sin_alpha)

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def head_scale(rotated_landmarks):
    head_hight = rotated_landmarks[33][1]-rotated_landmarks[27][1]
    return head_hight/180
	
if __name__ == "__main__":
    #this code will be executed only if we do not load this file as library

    #surface 3 pro camera 0
    #surface 4 pro camera 1
    cap = cv2.VideoCapture(0)

    
    #this part will be executed anyhow
    #is a intialization
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while True:
        _, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for i,face in enumerate(faces):
            landmarks = predictor(frame, face)
            landmarks = face_utils.shape_to_np(landmarks)
            for (x,y) in landmarks:
                cv2.circle(frame, (x, y), 1, (0, 0,255), -1)
            
            print('Roll angle: ',head_roll(landmarks),'Yaw angle: ',head_yaw(landmarks),'Pitch angle: ',head_pitch(landmarks))


        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
