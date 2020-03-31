
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
cap = cv2.VideoCapture(0)

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

def segment_eyes(frame, eye_left_landmarks, eye_right_landmarks, square=True):

    eye_left_pts    = np.array(list(map(lambda p: list(p.ravel()), eye_left_landmarks)))
    eye_right_pts   = np.array(list(map(lambda p: list(p.ravel()), eye_right_landmarks)))

    if square :
        l,r,t,b = bounding_box(eye_left_pts, frame.shape[1], frame.shape[0],0,10)
        #transfor exagon in a square
        left  = frame[t:b,l:r]
        mask  = np.ones_like(left)
        mask_l = mask.astype(np.bool)

        l,r,t,b = bounding_box(eye_right_pts, frame.shape[1], frame.shape[0],0,10)
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
    debug = False

    rows = img.shape[0]
    circles   = cv2.HoughCircles(np.uint8(img),cv2.HOUGH_GRADIENT,1,rows//10, param1=40, param2=24, minRadius=rows//8, maxRadius=rows//2)
    
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
            cv2.putText(new_tmp,str(i)+':'+str(v),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255)

    #sort depending on the average value 
    t = sorted(m.items(), key=lambda x: x[1])
    if len(t) == 0:
        return None
    
    if debug:
        cv2.imshow('detected circles deb ug',new_tmp)
   
    #return [x,y,r] of the most probable iris point
    return circles[0,t[0][0]]

def IPF_x(img,mask,x):
    return img[:,x].sum() /mask[:,x].sum()

def VPF_x(img, mask,x, IPF):
    return (np.abs(img[:,x] - IPF[x])).sum() / mask[:,x].sum() 

def IPF_y(img,mask,y):
    return img[y,:].sum() /mask[y,:].sum()

def VPF_y(img, mask, y, IPF):
    return (np.abs(img[y,:] - IPF[y])).sum() / mask[y,:].sum() 

def fit_iris_with_IPF(img, mask):
    debug          = False
    greatest_delta = True
    th_factor      = 0.05
    alpha          = 0.6
    delta          = 1
    #print(img.shape, mask.shape)
    cv2.imshow('mask',mask*255)
    cv2.imshow('eyes',img)
    
 
    IPF_values = list(map(lambda y  : IPF_y(img,mask,y)             , range(img.shape[0])))
    VPF_values = list(map(lambda y, : VPF_y(img,mask,y, IPF_values) , range(img.shape[0])))
    GPF_values = np.array(VPF_values) *np.float(alpha) - np.array(IPF_values) * np.float(1 - alpha) 

    if greatest_delta:
        IPF_delta  = (np.gradient(GPF_values,delta,axis=0))

        #1 step derivative
        #IPF_delta  = (list(map(lambda y,y_1: y_1-y, IPF_values[:-1], IPF_values[1:])))
 
        t = sorted(enumerate(IPF_delta), key=lambda x: x[1], reverse=True)
        first_y  = t[0][0]
        second_y = t[-1][0]

    else:
        threshold = (max(IPF_values)-min(IPF_values))* th_factor + min(IPF_values)
        flag = True
        first_y = 0
        second_y = 0
        for i in range(img.shape[0])  :
 
            if(IPF_values[i]   < threshold and flag ):
                flag = False
                first_y = i
            elif(IPF_values[i] > threshold and not flag):
                second_y = i
                break

    ############
    ############
    ############
    
    IPF_values = list(map(lambda y  : IPF_x(img,mask,y)             , range(img.shape[1])))
    VPF_values = list(map(lambda y, : VPF_x(img,mask,y, IPF_values) , range(img.shape[1])))
    GPF_values = np.float(alpha) * np.array(VPF_values) - np.float(1 - alpha) * np.array(IPF_values)


    if greatest_delta :
        IPF_delta  = (np.gradient(GPF_values,delta,axis=0))
 
        t = sorted(enumerate(IPF_delta), key=lambda x: x[1], reverse=True)
        first_x  = t[0][0]
        #t = list(filter(lambda e: (e[0]-first_x) > 0.1*img.shape[1] ,  t))
        second_x = t[-1][0]


    else:
        threshold = (max(IPF_values)-min(IPF_values))*th_factor + min(IPF_values)
        flag = True
 
        first_x=0
        second_x=0
 
        for i in range(img.shape[1]):
            if(IPF_values[i] < threshold and flag ):
                flag = False
                first_x = i
            elif(IPF_values[i] > threshold and not flag):
                second_x = i
                break
 
    
 
    if(debug):
        points = np.zeros(img.shape[1])
        points[first_x] = 255
        points[second_x] = 255
 
        fig,axs = plt.subplots(2)
        axs[0].plot(IPF_values,'r')
        
        if greatest_delta:
            axs[0].plot(IPF_delta ,'g')
        else :
            axs[0].plot([threshold for i in range(img.shape[1]) ])
        axs[0].plot(points)
        axs[1].imshow(img)
        plt.show()
 
        #PLOT
        cv2.line(img, (first_x,0 ),(first_x,img.shape[0]) , 255)
        cv2.line(img, (second_x,0),(second_x,img.shape[0]), 255)
        cv2.line(img, (0,first_y) ,(img.shape[1],first_y) ,255)
        cv2.line(img, (0,second_y),(img.shape[1],second_y),255)
 
        cv2.imshow('eye',img )
 
    x = (first_x+second_x)//2
    y = (first_y+second_y)//2
    r = np.abs(second_x-first_x)//2
 
    #return [x,y,r] of the most probable iris point
    return [x,y,r]

def rescale(img, scale_percent):
    
    width = int(img.shape[1] * scale_percent )
    height = int(img.shape[0] * scale_percent)
    
    dimR = (width, height)

    # resize image
    return cv2.resize(img, dimR, cv2.INTER_LINEAR) #.INTER_NEAREST)  #cv2.INTER_CUBIC )#interpolation=cv2.INTER_AREA )

def is_blinking(eye_top_landmark, eye_bottom_landmark, scale):
    H = math.sqrt((eye_top_landmark[0]-eye_bottom_landmark[0])** 2+(eye_top_landmark[1]-eye_bottom_landmark[1])**2)/scale
    if(H<16):
        return True

def iris_position(face_landmarks, eye_selector, detected_iris, img):
    debug = False                                                   # eye_selector = 0 -> left eye eye_selector = 1 -> right eye

    if(eye_selector == 0):
        eye_external_landmark = landmarks[36]
        eye_internal_landmark = landmarks[39]
        eye_top_landmark = landmarks[38]
        eye_bottom_landmark = landmarks[40]
    else:
        eye_external_landmark = landmarks[42]
        eye_internal_landmark = landmarks[45]
        eye_top_landmark = landmarks[43]
        eye_bottom_landmark = landmarks[47]

    D = math.sqrt((eye_external_landmark[0]-eye_internal_landmark[0])** 2+(eye_external_landmark[1]-eye_internal_landmark[1])**2)
    H = math.sqrt((eye_top_landmark[0]-eye_bottom_landmark[0])** 2+(eye_top_landmark[1]-eye_bottom_landmark[1])**2)

    eye_relative_position = project(np.asarray(eye_internal_landmark), np.asarray(eye_external_landmark), (detected_iris[:2]), D, H)

    R_d = eye_relative_position[0]/D      #ratio of position (all to the left = 0, all the way to the right = 1)
    R_h = eye_relative_position[1]/H      # ratio of position (all the way down = 0, all the way up = 1)
    return R_d, R_h

def head_roll(face_landmarks):                      #rotation along the axis perpendicular to the plane
    D = math.sqrt((landmarks[36][0]-landmarks[45][0])** 2+(landmarks[36][1]-landmarks[45][1])**2)
    return math.asin((landmarks[45][1]-landmarks[36][1])/D)*180/math.pi

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

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rot_mat,result

debug = False
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out_width  = 600
out_height = 200
out = cv2.VideoWriter("output.avi", fourcc, 10.0, (out_width, out_height))
head_hight = 0

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector(gray)

    for i,face in enumerate(faces):
        
       
        landmarks = predictor(frame, face)
        landmarks = face_utils.shape_to_np(landmarks)
        for (x,y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 0,255), -1)
        ##
        ##rotate the image
        matrx,gray  = rotate_image(gray,head_roll(landmarks))
        matrx,frame = rotate_image(frame,head_roll(landmarks))
        cv2.imshow('rotated', frame)
        #print(matrx)
        #rotate thge landmarks

        #print(np.array(landmarks).shape)
        #print(np.expand_dims(np.array(landmarks),axis=1).shape)

        landmarks_transf = cv2.transform(np.expand_dims(np.array(landmarks),axis=1), matrx)
        #print(landmarks_transf)
        landmarks = np.resize(landmarks_transf, landmarks.shape)
        #print(landmarks)
        ##
        extra_point_x = landmarks[21][0] + (landmarks[22][0]-landmarks[21][0])/2
        extra_point_y = landmarks[21][1] + (landmarks[22][1]-landmarks[21][1])/2
        cv2.line(frame, tuple(landmarks[36].ravel()), tuple(landmarks[45].ravel()), (0, 255, 0), thickness=3, lineType=8)
        cv2.line(frame, (int(extra_point_x), int(extra_point_y)), tuple(landmarks[57].ravel()), (255, 0, 0), thickness=3, lineType=8)
        #fit eyes
        #left 37-42 right 43-48

        if(debug):
            if(math.sqrt((landmarks[27][0]-landmarks[30][0]) ** 2+(landmarks[27][1]-landmarks[30][1])**2) > head_hight):
                print("__________________________________________________________________________")
                print(math.sqrt((landmarks[27][0]-landmarks[30][0])** 2+(landmarks[27][1]-landmarks[30][1])**2))
                print("__________________________________________________________________________")
        
        head_hight = math.sqrt((landmarks[27][0]-landmarks[33][0]) ** 2+(landmarks[27][1]-landmarks[33][1])**2)
        head_scale = head_hight/180  #empirical value

        

        (left, m_l),(right, m_r) = segment_eyes(gray, landmarks[36:42], landmarks[42:48])

        #detect iris:
        factor_magnification = 5
        left   = prep_fit_iris(left,factor_magnification)
        m_l    = rescale(m_l.astype(np.uint8),factor_magnification)
        c_left = fit_iris_with_IPF(left, m_l)

        right  = prep_fit_iris(right,factor_magnification)
        m_r    = rescale(m_r.astype(np.uint8),factor_magnification)
        c_right= fit_iris_with_IPF(right, m_r)

        #plot add 3rd channel to
        new_left = np.zeros([left.shape[0], left.shape[1], 3], np.uint8)
        new_left[:, :, 1] = left
        new_right = np.zeros([right.shape[0], right.shape[1], 3], np.uint8)
        new_right[:, :, 1] = right
        left_blink = is_blinking(landmarks[38], landmarks[40], head_scale)
        right_blink = is_blinking(landmarks[43], landmarks[47], head_scale)

        absolute_left_iris = [c_left[0]/factor_magnification+landmarks[36][0], c_left[1]/factor_magnification+landmarks[37][1]]
        cv2.circle(frame, (absolute_left_iris[0].ravel(), absolute_left_iris[1].ravel()) , 3, (0, 120, 135), -1)
        if(debug):
            print('Roll angle: ')
            print(head_roll(landmarks))
            print('Yaw angle: ')
            print(head_yaw(landmarks))
            print('Pitch angle: ')
            print(head_pitch(landmarks))
            print('Left Blink: ')
            print(left_blink)
            print('Right Blink: ')
            print(right_blink)

        left_illustration  = np.zeros((200,200,3))
        right_illustration = np.zeros((200,200,3)) 
        
        if(c_left is not None):
            if(left_blink != True):
                cv2.circle(new_left,(c_left[0],c_left[1]),c_left[2],(255,0,0),1)
                cv2.circle(new_left,(c_left[0],c_left[1]),2,(0,0,255),2)
                iris_pose_left = iris_position(landmarks, 0, absolute_left_iris, left)
                #print(iris_pose_left)
                p = (int(iris_pose_left[0]*left_illustration.shape[1]), int(iris_pose_left[1]*left_illustration.shape[0]) )
                r = left_illustration.shape[1]//4
                cv2.circle(left_illustration, p , r ,(255,0,0), 1)
            
        if(c_right is not None):
            if(right_blink != True):
                cv2.circle(new_right,(c_right[0],c_right[1]),c_right[2],(255,0,0),1)
                cv2.circle(new_right,(c_right[0],c_right[1]),2,(0,0,255),2)
                iris_pose_right= iris_position(landmarks, 1, (c_right[0]/factor_magnification+landmarks[42][0], c_right[1]/factor_magnification+landmarks[43][1]), right)
                #print(iris_pose_right)
                p = (int(iris_pose_right[0]*right_illustration.shape[1]), int(iris_pose_right[1]*right_illustration.shape[0]))
                r = right_illustration.shape[1]//4
                cv2.circle(right_illustration, p , r , (255,0,0), 1)

        frame_illustration =  stack(left_illustration,right_illustration)
        cv2.imshow('position',frame_illustration )

        frame_out = stack(new_left,new_right)
        cv2.imshow('Detected circles',frame_out )

        if(right_blink and left_blink):
            print("Blink detected!")
        elif(right_blink or left_blink):
            print("Wink detected!")

        frame_out = cv2.resize(frame_out, (out_width, out_height))
        out.write(frame_out)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
out.release()
