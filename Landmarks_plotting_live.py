
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
import matplotlib
import matplotlib.pyplot as plt
import gaze_estimation 

#surface 3 pro camera 0
#surface 4 pro camera 1
cap = cv2.VideoCapture(0)

debug = False

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out_width  = 600
out_height = 200
out = cv2.VideoWriter("output.avi", fourcc, 10.0, (out_width, out_height))

while True:
    _, frame = cap.read()

    for p,estimate in gaze_estimation.process(frame):
        print(p, estimate)
    

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
out.release()
