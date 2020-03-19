
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import cv2
import imutils
import time
import dlib

cap = cv2.VideoCapture(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        shape = predictor(frame, face)
        shape = face_utils.shape_to_np(shape)
        for (x,y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        cv2.line(frame, tuple(shape[36].ravel()), tuple(shape[45].ravel()), (0, 255, 0), thickness=3, lineType=8)
        cv2.line(frame, tuple(shape[27].ravel()), tuple(shape[57].ravel()), (255, 0, 0), thickness=3, lineType=8)
        landmarks = predictor(gray, face)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
