
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

#surface 3 pro camera 0
#surface 4 pro camera 1
cap = cv2.VideoCapture(1)

def exagon_padding(pts, padx, pady):
    pts[0,0]-=padx
    pts[1,1]-=pady
    pts[2,1]-=pady
    pts[3,0]+=padx
    pts[4,1]+=pady
    pts[5,1]+=pady
    return pts

def bounding_box(pts,  width, height, padx=0, pady=0,):
    left  = max(min(pts[:,0]) - padx, 0)
    right = min(max(pts[:,0]) + padx, width)
    top   = max(min(pts[:,1]) - pady,0)
    bottom= min(max(pts[:,1]) + pady, height)
   
    return left,right,top,bottom

def extract_polyline(img, poly_pts):
    l,r,t,b   = bounding_box(poly_pts, img.shape[1], img.shape[0])
    sub_img   = np.array(img[t:b+1,l:r+1])

    poly_pts  = np.array(list(map(lambda p: [p[0]-l,p[1]-t], poly_pts)))

    mask      = np.zeros_like(sub_img)
    
    cv2.fillConvexPoly(mask, poly_pts,  1)
    mask = mask.astype(np.bool)
 
    out       = np.zeros_like(sub_img)
    out[mask] = sub_img[mask]
    

    return out, mask

def segment_eyes(frame, eye_left_landmarks, eye_right_landmarks, square=False):

    eye_left_pts    = np.array(list(map(lambda p: list(p.ravel()), eye_left_landmarks)))
    eye_right_pts   = np.array(list(map(lambda p: list(p.ravel()), eye_right_landmarks)))

    if square :
        l,r,t,b = bounding_box(eye_left_pts, frame.shape[1], frame.shape[0],10,10)
        #transfor exagon in a square
        left  = frame[t:b,l:r]
        mask  = np.ones_like(left)
        mask_l = mask.astype(np.bool)

        l,r,t,b = bounding_box(eye_right_pts, frame.shape[1], frame.shape[0],10,10)
        #transfor exagon in a square
        right  = frame[t:b,l:r]
        mask   = np.ones_like(right)
        mask_r = mask.astype(np.bool)
    else:
        pts   = exagon_padding(eye_left_pts, 5,5)
        left, mask_l   = extract_polyline(frame, pts)

        pts   = exagon_padding(eye_right_pts,5,5)
        right, mask_r = extract_polyline(frame, pts)

    return (left, mask_l), (right, mask_r)

def prep_fit_iris(img, rscale_factor=5):
    
    img     = cv2.equalizeHist(img)
    img     = rescale(img, rscale_factor)
    img     = cv2.GaussianBlur(img,(3,3),1)
    return img

def fit_iris(img):
    debug = True

    rows = img.shape[0]
    circles   = cv2.HoughCircles(np.uint8(img),cv2.HOUGH_GRADIENT,1,rows//10, param1=70, param2=28, minRadius=rows//8, maxRadius=rows//2)

    if circles is None :
        return

    new_tmp = np.zeros([img.shape[0], img.shape[1],3], np.uint8)
    new_tmp[:,:,1] = img

    circles = np.uint16(np.around(circles))
    m = {}
    for i,c in enumerate(circles[0,:]):
        # draw the outer circle
        x = c[0]
        y = c[1]
        r = c[2]
        # compute cumulative value in that circle
        
        v = get_mean_value_inside_circle(img,x,y,r)
        if v > 70:
            continue

        m[i] = v
        
        if debug:
            cv2.circle(new_tmp,(x,y),r,(0,255,0),1)
            #draw the center of the circle
            cv2.circle(new_tmp,(x,y),2,(0,0,255),2)
            #and average pixel value
            cv2.putText(new_tmp,str(i),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255)

    #sort depending on the average value 
    t = sorted(m.items(), key=lambda x: x[1])
    if len(t) == 0:
        return None
    
    if debug:
        cv2.imshow('detected circles',new_tmp)
   
    #return [x,y,r] of the most probable iris point
    return circles[0,t[0][0]]

def IPF_x(img,mask,x):
    return img[:,x].sum() /mask[:,x].sum()

def IPF_y(img,mask,y):
    return img[y,:].sum() /mask[y,:].sum()

def fit_iris_with_IPF(img, mask):
    
    #print(img.shape, mask.shape)
    cv2.imshow('mask',mask*255)


    IPF_values = list(map(lambda y: IPF_y(img,mask,y), range(img.shape[0])))
    IPF_delta  = np.abs(np.gradient(IPF_values,axis=0))

    '''plt.figure()
    plt.plot(IPF_values,'r')
    plt.plot(IPF_delta ,'g')
    plt.show()'''

    t = sorted(enumerate(IPF_delta), key=lambda x: x[1], reverse=True)
    first_x  = t[0][0]
    second_x = t[2][0]
    cv2.line(img, (first_x,0 ),(first_x,img.shape[0]) , 255)
    cv2.line(img, (second_x,0),(second_x,img.shape[0]), 255)

    IPF_values = list(map(lambda x: IPF_x(img,mask,x), range(img.shape[1])))
    IPF_delta  = np.abs(np.gradient(IPF_values,axis=0))

    '''plt.figure()
    plt.plot(IPF_values,'r')
    plt.plot(IPF_delta ,'g')
    plt.show()
    '''
    t = sorted(enumerate(IPF_delta), key=lambda x: x[1], reverse=True)
    first_y  = t[0][0]
    second_y = t[2][0]
    cv2.line(img, (0,first_y) ,(img.shape[1],first_y) ,255)
    cv2.line(img, (0,second_y),(img.shape[1],second_y),255)

    cv2.imshow('eye',img )
    #return [x,y,r] of the most probable iris point
    return None

def rescale(img, scale_percent):
    
    width = int(img.shape[1] * scale_percent )
    height = int(img.shape[0] * scale_percent)
    
    dimR = (width, height)

    # resize image
    return cv2.resize(img, dimR, interpolation=cv2.INTER_AREA )


# eye_selector = 0 -> left eye eye_selector = 1 -> right eye
def iris_position(face_landmarks, eye_selector, detected_iris):
    if(eye_selector == 0):
        eye_external_landmark = landmarks[36]
        eye_internal_landmark = landmarks[39]
        eye_top_landmark = landmarks[38]
        eye_bottom_landmark = landmarks[40]
    else:
        eye_external_landmark = landmarks[45]
        eye_internal_landmark = landmarks[42]
        eye_top_landmark = landmarks[43]
        eye_bottom_landmark = landmarks[47]

    D = math.sqrt((eye_external_landmark[0]-eye_internal_landmark[0])** 2+(eye_external_landmark[1]-eye_internal_landmark[1])**2)
    H = math.sqrt((eye_top_landmark[0]-eye_bottom_landmark[0])** 2+(eye_top_landmark[1]-eye_bottom_landmark[1])**2)

    eye_relative_position = project(np.asarray(eye_internal_landmark), np.asarray(eye_external_landmark), (detected_iris[:2]), D, H)

    R_d = eye_relative_position[0]/D      #ratio of position (all versus the nose = 0, all the way out = 1)
    R_h = eye_relative_position[1]/H        # ratio of position (all the way down = 0, all the way up = 1)
    return R_d, R_h


def head_roll(face_landmarks):                      #rotation along the axis perpendicular to the plane
    D = math.sqrt((landmarks[36][0]-landmarks[45][0])** 2+(landmarks[36][1]-landmarks[45][1])**2)
    return math.asin((landmarks[45][1]-landmarks[36][1])/D)

def head_pitch(face_landmarks):                     # rotation along the orizontal axis TO BE CALIBRATED
    extra_point_x = face_landmarks[21][0]+(face_landmarks[22][0]-face_landmarks[21][0])/2
    extra_point_y = face_landmarks[21][1]+(face_landmarks[22][1]-face_landmarks[21][1])/2

    D = math.sqrt((extra_point_x-face_landmarks[57][0])** 2+(extra_point_y - face_landmarks[57][1])**2)
    centre_point = line_intersection((face_landmarks[36], face_landmarks[45]), ((extra_point_x, extra_point_y), face_landmarks[57]))
    bottom_to_centre = math.sqrt(((face_landmarks[57][0]-centre_point[0]) ** 2)+(face_landmarks[57][1]-centre_point[1])**2)
    return math.asin(bottom_to_centre/D)

def head_yaw(face_landmarks):                       #rotation along the vertical axis TO BE CALIBRATED
    D = math.sqrt((face_landmarks[36][0]-face_landmarks[45][0])** 2+(face_landmarks[36][1]-face_landmarks[45][1])**2)
    centre_point = line_intersection((face_landmarks[36], face_landmarks[45]), (face_landmarks[27], face_landmarks[57]))
    left_to_centre = math.sqrt(((face_landmarks[36][0]-centre_point[0]) ** 2)+(face_landmarks[36][1]-centre_point[1])**2)
    return math.asin(left_to_centre/D)

def project(line_point_1, line_point_2, point, D, H):
    a = point-line_point_1
    A = math.sqrt((line_point_1[0]-point[0])**2 +(line_point_1[1]-point[1])**2)
    d = line_point_2-line_point_1
    cos_alpha = np.dot(a, d)/(D*A)
    sin_alpha = np.sin(np.arccos(cos_alpha))
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

def get_mean_value_inside_circle(img,xc,yc,radius):
    
    y,x = np.ogrid[0:img.shape[0], 0:img.shape[1]]
    mask = ( ( (x-xc)**2 + (y-yc)**2 ) <= radius**2) 
    val = img[mask].mean()

    return val

def stack(one,two):
    tmp = np.zeros([max(one.shape[0],two.shape[0])*2, max(one.shape[1],two.shape[1])*2, 3], np.uint8)
    tmp[0:one.shape[0], 0:one.shape[1],:] = one
    tmp[0:two.shape[0], one.shape[1]:one.shape[1]+two.shape[1],:] = two
    return tmp

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out_width  = 600
out_height = 200
out = cv2.VideoWriter("output.avi", fourcc, 10.0, (out_width, out_height))


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
        extra_point_x = landmarks[21][0] + (landmarks[22][0]-landmarks[21][0])/2
        extra_point_y = landmarks[21][1] + (landmarks[22][1]-landmarks[21][1])/2
        cv2.line(frame, tuple(landmarks[36].ravel()), tuple(landmarks[45].ravel()), (0, 255, 0), thickness=3, lineType=8)
        cv2.line(frame, (int(extra_point_x), int(extra_point_y)), tuple(landmarks[57].ravel()), (255, 0, 0), thickness=3, lineType=8)
        #fit eyes
        #left 37-42 right 43-48

        (left, m_l),(right, m_r) = segment_eyes(gray, landmarks[36:42], landmarks[42:48])

        #detect iris:
        left   = prep_fit_iris(left,5)
        m_l    = rescale(m_l.astype(np.uint8),5)
        c_left = fit_iris(left)

        right  = prep_fit_iris(right,5)
        m_r    = rescale(m_r.astype(np.uint8),5)
        c_right= fit_iris(right)

        #print('Roll angle: ')
        #print(head_roll(landmarks))
        #print('Yaw angle: ')
        #print(head_yaw(landmarks))
        #print('Pitch angle: ')
        #print(head_pitch(landmarks))
        
        iris_pose_left = iris_position(landmarks, 0, (c_left[0]/5+landmarks[36][0], c_left[1]/5+landmarks[37][1]))
        iris_pose_right= iris_position(landmarks, 1, (c_right[0]/5+landmarks[42][0], c_right[1]/5+landmarks[43][1]))

        #plot add 3rd channel to 
        new_left = np.zeros([left.shape[0], left.shape[1],3],np.uint8)
        new_left[:,:,1] = left
        new_right = np.zeros([right.shape[0], right.shape[1],3], np.uint8)
        new_right[:,:,1] = right

        if(c_left is not None):
            cv2.circle(new_left,(c_left[0],c_left[1]),c_left[2],(255,0,0),1)
            cv2.circle(new_left,(c_left[0],c_left[1]),2,(0,0,255),2)
        if(c_right is not None):
            cv2.circle(new_right,(c_right[0],c_right[1]),c_right[2],(255,0,0),1)
            cv2.circle(new_right,(c_right[0],c_right[1]),2,(0,0,255),2)
        
        frame_out = stack(new_left,new_right)
        cv2.imshow('detected circles',frame_out )
        
        frame_out = cv2.resize(frame_out, (out_width, out_height))
        out.write(frame_out)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
out.release()
