import cv2
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import VideoStream

import dlib
import head
from head import head_roll

class Rotator:
    def __init__(self,rotation_point, angle):
        #image_center = tuple(np.array(img.shape[1::-1]) / 2)
        
        #self.original_image          = img
        self.rotation_matrix         = cv2.getRotationMatrix2D(rotation_point, angle, 1.0)
        self.reverse_rotation_matrix = cv2.getRotationMatrix2D(rotation_point, -angle, 1.0)
        #self.rotated_image           = cv2.warpAffine(img, self.rotation_matrix, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    def __getitem__(self,p):
        return self.rotated_image[p]
    
    def transform_point(self,pts):
        pts_exp = np.expand_dims(np.array(pts),axis=1)
        pts_transf_exp = cv2.transform(pts_exp, self.rotation_matrix)
        return np.resize(pts_transf_exp, pts.shape)

    def reverse_transform_point(self,pts):
        pts_exp = np.expand_dims(np.array(pts),axis=1)
        pts_transf_exp = cv2.transform(pts_exp, self.reverse_rotation_matrix)
        return np.resize(pts_transf_exp, pts.shape)
    
    def transform(self,img):
        return cv2.warpAffine(img, self.rotation_matrix, img.shape[1::-1], flags=cv2.INTER_LINEAR) 

    def reverse_transform(self,img):
        return  cv2.warpAffine(img, self.reverse_rotation_matrix, img.shape[1::-1], flags=cv2.INTER_LINEAR)


if __name__ == "__main__":
    #this code will be executed only if we do not load this file as library

    #surface 3 pro camera 0
    #surface 4 pro camera 1
    cap = cv2.VideoCapture(0)

    
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
            
            
            roll_angle      = head_roll(landmarks)
            image_center    = tuple(np.array(frame.shape[1::-1]) / 2)
            rot             = Rotator(image_center,roll_angle)
            rotated_frame   = rot.transform(frame)


            cv2.imshow("rotated frame "+str(i), rotated_frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
