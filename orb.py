import cv2
import numpy as np
from numpy import linalg as LA
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize,remove_small_objects, area_closing

cap = cv2.VideoCapture("CCTV.mp4")
# Create ORB object
feature_extractor = cv2.ORB_create(500)
# FLANN parameters used to determine the match
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
# take a frame and extract the first parameters

keypoints, descriptor  = None, None
frame = None


while (cap.isOpened()):
    frame_last = frame
    _, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # find the keypoints with ORB
    keypoints_last, descriptor_last  = keypoints, descriptor 
    keypoints, descriptor            = feature_extractor.detectAndCompute(gray,None)
    
    # keypoints in the last frame
    #img= cv2.drawKeypoints(gray,kp1,None,(255,0,0),4)
    #cv2.imshow("Frame", img)    
    if frame_last is None or descriptor_last is None:
        continue



    matches = matcher.knnMatch(np.asarray(descriptor,np.float32),np.asarray(descriptor_last,np.float32),2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]


    # ratio test as per Lowe's paper and plot
    img = frame  
    for i,(m,n) in enumerate(matches):
        p1 = keypoints[m.trainIdx].pt
        p2 = keypoints_last[m.trainIdx].pt
        norm = np.sqrt( (p1[0] - p2[0])** 2 + (p1[1] - p2[1])**2 )
        
        if (m.distance < 0.7*n.distance and
            norm > 50 and  norm < 200 ):
           
            cv2.circle(img,tuple(np.int32(p1)), 4, (255,0,0))

    cv2.imshow("Frame", img)    
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
