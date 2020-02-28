import cv2
import numpy as np

verbose = True

cap = cv2.VideoCapture("CCTV.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#avi conversion and scaling for an easier computation
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280, 720))

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    #frame difference
    diff = cv2.absdiff(frame1, frame2)
    #grayscale compression
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #gaussian blure to remove small areas
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if verbose :
        cv2.imshow("Diff", diff)
        cv2.imshow("Gray", gray)
        cv2.imshow("Blur", blur)

    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=4)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if verbose :
        cv2.imshow("Countour", dilated)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 1200:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.drawContours(frame1, contours, -1, (203, 50, 52), 2)

    image = cv2.resize(frame1, (1280, 720))
    out.write(image)
    cv2.imshow("Feed", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
out.release()
