import json
import numpy as np
import os
import cv2
import threading
#from PIL import Image

directory_images = 'iris_position_test_directory'

def image2json_string(image):
    return json.dumps(np.array(image).tolist())

def json_string2image(string):
    return (np.array(json.loads(string), dtype='uint8'))

def peek_point(event,x,y,flags,param):
    trigger, list_pts = param 
    if event == cv2.EVENT_LBUTTONDBLCLK:
        list_pts.append((x,y))
        trigger.set()

def fill_json_with_data_from_user(json_data):
    for root, dirs, files in os.walk(directory_images):
        files = filter(lambda name: name.lower().endswith(('.png', '.jpg', '.jpeg')) and name not in json_data, files)
        
        for filename in files:
            abs_filename = os.path.join(root,filename)
            img = cv2.imread(abs_filename)

            #create event for continuation
            trigger = threading.Event()
            #list where pts would be put
            list_pts = []
            cv2.putText(img,  "In order indicate left_extreme_1, centre, left_extreme_2, then pass to right", (20,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (209, 80, 0, 255), 1) 
            cv2.imshow('image_to_be_annotated', img)
            cv2.setMouseCallback('image_to_be_annotated',peek_point, param=(trigger,list_pts))
            
            #peek six points
            while len(list_pts) < 6:

                while( not trigger.wait(0.30)):
                    if cv2.waitKey(20) == 27:
                        #eliminate last point
                        if len(list_pts) > 0 :
                            del list_pts[-1]
                        else:
                            exit(1)
                        
                #draw new point
                cv2.circle(img,list_pts[-1],2,(0,0,255),-1)
                #reset trigger
                trigger.clear()
                #repaint
                cv2.imshow('image_to_be_annotated', img)
                

            cv2.destroyWindow('image_to_be_annotated')

            # print(list_pts) # peek data
            data_for_json = {}
            data_for_json['left_eye_extreme1']  = list_pts[0]
            data_for_json['left_eye_extreme2']  = list_pts[2]
            data_for_json['left_eye_centre']    = list_pts[1]
            data_for_json['right_eye_extreme1'] = list_pts[3]
            data_for_json['right_eye_extreme2'] = list_pts[5]
            data_for_json['right_eye_centre']   = list_pts[4]
            # fill json 
            json_data[filename] = data_for_json

if __name__ == "__main__":
    json_data = {}
    json_path = os.path.join(directory_images,'json.json')
    #open already collected data
    if os.path.exists(json_path):
        json_fd     = open(json_path)
        json_data   = json.loads(json_fd.read())
        json_fd.close()
    #integrate with new images
    fill_json_with_data_from_user(json_data)
    # overwrite file   
    json_fd     = open(json_path, 'w')
    json_fd.write(json.dumps(json_data))
    json_fd.close()
