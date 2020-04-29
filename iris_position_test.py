import json
import numpy as np
import os
import cv2
import threading
import iris_position as ir_pos


directory_images    = 'iris_position_test_directory'
left_eye_extreme1   = 'left_eye_extreme1'  
left_eye_extreme2   = 'left_eye_extreme2'  
left_eye_centre     = 'left_eye_centre'    
right_eye_extreme1  = 'right_eye_extreme1' 
right_eye_extreme2  = 'right_eye_extreme2' 
right_eye_centre    = 'right_eye_centre'   
delay_to_see_debug  = 5000
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
            data_for_json[left_eye_extreme1     ]= list_pts[0]
            data_for_json[left_eye_centre       ]= list_pts[2]
            data_for_json[left_eye_extreme2     ]= list_pts[1]
            data_for_json[right_eye_extreme1    ]= list_pts[3]
            data_for_json[right_eye_extreme2    ]= list_pts[5]
            data_for_json[right_eye_centre      ]= list_pts[4]
            # fill json 
            json_data[filename] = data_for_json

def test_data(json_data, debug=False):
    err_left_abs    = 0
    err_right_abs   = 0
    for filename in json_data:
        abs_filename = os.path.join(directory_images,filename)
        img = cv2.imread(abs_filename)

        iris_info = ir_pos.irides_position_form_video(img)

        
        flag_first = True
        for p in iris_info:
            if not flag_first:
                print("WARNING:" , filename, "\t multiple faces detected")
            abs_left, abs_right, rel_left, rel_right = p
            centre_left_predicted   = np.array(abs_left[:2])
            centre_right_predicted  = np.array(abs_right[:2])
            centre_left_expected    = json_data[filename][left_eye_centre  ]
            centre_right_expected   = json_data[filename][right_eye_centre ] 

            if debug:
                #green groud truth
                tmp = cv2.circle(img, tuple(centre_left_expected                    ),1,(0,255,0),1)
                tmp = cv2.circle(tmp, tuple(centre_right_expected                   ),1,(0,255,0),1)
                #red predicted
                tmp = cv2.circle(tmp, tuple(centre_left_predicted.astype(np.int)    ),1,(0,0,255),1)
                tmp = cv2.circle(tmp, tuple(centre_right_predicted.astype(np.int)   ),1,(0,0,255),1)
                cv2.imshow('result',tmp)
                cv2.waitKey(delay_to_see_debug)
            
            err_left_abs    += np.linalg.norm(centre_left_predicted-centre_left_expected) 
            err_right_abs   += np.linalg.norm(centre_right_predicted-centre_right_expected) 
            flag_first       =False

        if flag_first:
            print("WARNING:" , filename, "\t no face detected")

    sample_left_abs_err     = err_left_abs/len(json_data.keys())
    sample_right_abs_err    = err_right_abs/len(json_data.keys())
    print('Sample error left: ',sample_left_abs_err,'\t right:', sample_right_abs_err)


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

    test_data(json_data, True)
    
