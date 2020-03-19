import cv2
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize,remove_small_objects, area_closing

cap = cv2.VideoCapture("CCTV.mp4")

subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=10, detectShadows=False)
shadow_number = subtractor.getShadowValue()

while (cap.isOpened()):
    _, frame = cap.read()

    mask = subtractor.apply(frame)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = np.where(mask == shadow_number, 0, mask)

    clear = remove_small_objects(mask, 256)
    #clear = area_closing(clear, 128)
    clear = np.where(clear , 255, 0)

    # label image regions
    label_image = label(clear)
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            clear[minr, minc:maxc-1] = 127
            clear[minr:maxr-1, minc] = 127
            clear[maxr-1, minc:maxc-1] = 127
            clear[minr:maxr-1, maxc-1] = 127

    #print(out.shape)
    cv2.imshow("Frame", frame)
    cv2.imshow('segmented',  np.float32(clear))
    

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
