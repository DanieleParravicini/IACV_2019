import cv2
import numpy as np
from skimage import data, filters

verbose = True

cap = cv2.VideoCapture("CCTV.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#avi conversion and scaling for an easier computation
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter("output.avi", fourcc, 10.0, (1280, 720))

ret, frame1 = cap.read()
ret, frame2 = cap.read()

def fill_holes(img2fill ):
    #remove holes in the image img2fill
    img = img2fill.copy()
    
    #create a mask that is 2 pixels greater in every dimension
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0) which is expected to be 0
    # if this attempt works you will fill the area external to the 
    # figures stored in closed image.
    # The result would have 1 whether it was a figure
    # or it makes part of the foreground.
    # In other words what has kept 0 value is a hole
    # What you just have to do is to set those bit 1 
    # in the original image. This can be done easily using an or operation
    if img[0,0] != 0:
        print("WARNING: Filling something you shouldn't")
    cv2.floodFill(img, mask, (0,0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(img)
    im_out = img2fill | im_floodfill_inv
    if verbose :
        cv2.imshow('filled', np.float32(im_out))

    return im_out

while cap.isOpened():
    #frame difference
    diff = cv2.absdiff(frame1, frame2)
    #to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #gaussian blur to remove small areas
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if verbose :
        cv2.imshow("Diff", diff)
        cv2.imshow("Gray", gray)
        cv2.imshow("Blur", blur)
    #filter pixels which are higher than a given threshold
    threshold = 20
    value_if_above_threshold = 255
    
    _, thresholded = cv2.threshold(blur, threshold, value_if_above_threshold, 
       cv2.THRESH_BINARY)
    #close those pixels to recover connection with component which are not that far apart
    #dilate - fill - erode
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    small_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    closed = cv2.dilate(thresholded, ellipse, iterations=2)
    closed = fill_holes(closed)
    closed = cv2.erode(closed, ellipse, iterations=2)
    #erode dilate to eliminate small connections
    closed = cv2.erode(closed, small_ellipse, iterations=2)
    closed = cv2.dilate(closed, small_ellipse, iterations=2)
    
    
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #draw all contours
    blue = (203, 50, 52)
    cv2.drawContours(frame1, contours, -1, blue , 2)
    #draw interseting box
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        green = (0, 255, 0)
        red = (0,0,255)
        #skip contours which are less than 1200 pixels
        if cv2.contourArea(c) < 1200:
            continue
        #draw a rectangle
        cv2.rectangle(frame1, (x, y), (x+w, y+h), green , 2)
       
    #force the frame to a fixed height/width for saving
    image = cv2.resize(frame1, (1280, 720))
    out.write(image)
    cv2.imshow("Feed", frame1)

    #move to the next frame 
    #after moving the most recent frame 
    #into a temporary value (frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    #if esc is tapped exit.
    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
out.release()