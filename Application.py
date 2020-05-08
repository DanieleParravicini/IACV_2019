import tkinter as tk
import time
import threading
import ctypes
import numpy as np
import iris_position as ir_pos
import cv2
import settings 
import mouse
from iris_position_tracker import iris_position_tracker


calibration_done            = False
debug                       = False
useAbsPosition              = True
calibration_eye_point_left  = []
calibration_eye_point_right = []
calibration_point           = []
#x_l = [788.83569323, -1864.62986779, 7511.81004873, -20722.69084935, 4419.24471401, -1900.23835635, -5613.35123008, 85767.55041025, -46978.37603168, -23478.1630365]  # calibration parameters for me to test overwritten by calibration
#x_r = [976.19248026, 4989.40397841, 4598.99944016, -20727.05497126, -3023.62783183, 10482.96314502, -14402.16369339, -30344.84762274, 1588.7347783, 32797.23808307]
x_l = [4345.72378965, -13689.7759815, -9581.0823355, 46100.87408067, -16833.59181134, -43571.70242215, 22602.53804898, 157015.11947266, -28691.93582351, -100083.86167376]
x_r = [3.30882637e-01, 5.82059229e+02, -7.11895683e+02, 1.51578799e+00, -1.44032318e+00, -5.29422288e+02, 6.51063269e+02, 6.20962177e-01, 4.73292176e-01, -1.46584279e+00]

def build_iris_param_array(rel_pose):
    return np.array([1, rel_pose[0], rel_pose[1], rel_pose[0]*rel_pose[1], rel_pose[0]**2, rel_pose[1]**2])

def build_unknown_array(x):
    return np.array([[x[0], x[1], x[2], x[3], x[4], x[0]], [x[0], x[5], x[6], x[7], x[8], x[9]]])

class Home:
    def __init__(self, master):
        self.master = master
        self.master.geometry("400x400")
        self.frame = tk.Frame(self.master)
        if(calibration_done == False):
            tk.Label(self.frame, text="Calibration required before use:").pack(
                side="top")
        self.butnew("Calibration", "2", Calibration)
        tk.Button(self.frame, text = "Gaze Mouse", command = self.mouseControl).pack()
        self.butnew("Precision Test", "3", Precision)
        self.frame.pack()

    def mouseControl(self):
        self.master.iconify()
        #Todo: here i compute and control the mouse position.
        X_l = build_unknown_array(x_l)
        X_r = build_unknown_array(x_r)
       
        iris_tracker = iris_position_tracker(settings.camera,nr_samples_per_read=5,err_abs=6.5, err_rel=6.5, debug=False)
        
        gaze_r = None
        gaze_l = None
        for _ in range(100):
            abs_l, abs_r, rel_l, rel_r = next(iris_tracker)
            while all(e is None for e in [rel_l, rel_r]):
            
                abs_l, abs_r, rel_l, rel_r = next(iris_tracker)

            v_l , v_r = rel_l, rel_r      
            print(v_l, v_r)
            if v_l is not None:
                iris_l_param = build_iris_param_array(v_l)
                gaze_l = X_l.dot(iris_l_param)
            if v_r is not None:
                iris_r_param = build_iris_param_array(v_r)   
                gaze_r = X_r.dot(iris_r_param)
            

            if v_l is not None and v_r is not None:
                gaze = (((gaze_l+gaze_r)/2))
            elif v_l is not None:
                gaze = gaze_l
            elif v_r is not None:
                gaze = gaze_r
            else:
                gaze = None

            if(gaze is not None and np.all(gaze >=0)):
                mouse.move(gaze[0], gaze[1], absolute=True, duration=0)
                print('Left: ' , gaze_r)
                print('Right: ', gaze_l)
                print('Avg:'   , gaze)

        iris_tracker.close()    

    def butnew(self, text, number, _class):
        tk.Button(self.frame, text=text,
                  command=lambda: self.new_window(number, _class)).pack()

    def new_window(self, number, _class):
        self.new = tk.Toplevel(self.master)
        _class(self.new, number)
class Calibration:
    def __init__(self, master, number):
        self.master = master
        self.master.attributes('-fullscreen', True)
        self.frame = tk.Frame(self.master)
        self.explanation = tk.Text(self.frame, height="2", font="Helvetica")
        self.explanation.pack()
        self.explanation.insert(tk.END, "You have to click in order on the red dot in sequence and waiting 5 seconds\n before move to the next!")
        self.back = tk.Button(self.frame, text=f"<- Quit Calibration!", fg="red", command=self.close_window)
        self.back.pack()
        self.calibration_points()
        self.frame.pack()

    def create_circle(self, x, y, canvas):  # center coordinates, radius
        r = 5
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        return canvas.create_oval(x0, y0, x1, y1, fill="red", activefill = 'orange')

    def drawCircle(self, x, y, row, column):
        self.circular_button = self.create_circle(x, y, self.canvas)
        button_number = column+(row-1)*3
        self.canvas.create_text(x,y-15, text=f"{button_number}")
        # self.clicked just for testing but than it will call calibrate
        self.canvas.tag_bind(self.circular_button, "<Button-1>", lambda event, circle_button = self.circular_button: self.calibrate(event, button_number))

    def clicked(self, event, id_button):        #function used just for debug
        self.canvas.itemconfig(self.circular_button, fill="green")
        # to be investigate how to change color over circle presssed
        self.explanation.delete('1.0', tk.END)
        self.explanation.insert(tk.END, f"You have pressed {id_button}th point!")
        print(id_button)

    def calibration_points(self):
        self.canvas = tk.Canvas(self.master)
        user32 = ctypes.windll.user32
        padding = 100
        x = (user32.GetSystemMetrics(0)-2*padding)/2
        y = (user32.GetSystemMetrics(1)-2*padding)/2
        for row in range(3):
            for column in range(3):
                self.drawCircle(x*column+padding, y*row+padding, row+1, column+1)
                point = (x*column+padding, y*row+padding)
                calibration_point.append(point)
        self.canvas.pack(fill=tk.BOTH, expand=1)

    def calibrate(self, event, id_button):
        self.explanation.delete('1.0', tk.END)
        self.explanation.insert(tk.END, f"You have pressed {id_button}th point!\n Calibration started!")
        
    
        iris_tracker = iris_position_tracker(settings.camera,nr_samples_per_read=8,err_abs=6.5, err_rel=6.5, debug=False)
        abs_l, abs_r, rel_l, rel_r = next(iris_tracker)
        while any(e is None for e in [rel_l, rel_r ]):
           
            abs_l, abs_r, rel_l, rel_r = next(iris_tracker)

        v_l , v_r = rel_l, rel_r
              
        print(v_l, v_r)
        calibration_eye_point_left.append(np.array([1+v_l[1]**2, v_l[0], v_l[1], v_l[0]*v_l[1], v_l[0]**2, 0, 0, 0, 0, 0]))
        calibration_eye_point_left.append(np.array([1, 0, 0, 0, 0, v_l[0], v_l[1], v_l[0]*v_l[1], v_l[0]**2, v_l[1]**2]))
        calibration_eye_point_right.append(np.array([1+v_r[1]**2, v_r[0], v_r[1], v_r[0]*v_r[1], v_r[0]**2, 0, 0, 0, 0, 0]))
        calibration_eye_point_right.append(np.array([1, 0, 0, 0, 0, v_r[0], v_r[1], v_r[0]*v_r[1], v_r[0]**2, v_r[1]**2]))
        iris_tracker.close()
        if(id_button < 9):
            self.explanation.delete('1.0', tk.END)
            self.explanation.insert(tk.END, f"Click on the {id_button+1}th point!")            #consider to change the visual appearance of the dot to give a feedback  to user
        else:
            self.explanation.delete('1.0', tk.END)
            self.explanation.insert(tk.END, f"Calibration Ended!")
            self.compute_parameters()


    def compute_parameters(self):
        #devo risolvere un sistema di 9*2 equazioni in (6*2-3)*2 incognite
        #for each point i call a function that calls gaze position and returns the actual eye position and i average out this value.
        # A=[[1+v_y^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #   [1+v_x^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #   [1+v_x^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #   [1+v_x^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #   [1+v_x^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #   [1+v_x^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #   [1+v_x^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #   [1+v_x^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #   [1+v_x^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #x=[a_0, a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4, b_5]]
        #B=[s_x,s_y,s_x,s_y,s_x,s_y,s_x,s_y,s_x,s_y,s_x,s_y,s_x,s_y,s_x,s_y,s_x,s_y]

        global x_l
        global x_r

        B = np.asarray(calibration_point).flatten()
        B=B.astype(int)

        #left eye
        A_l = np.vstack(calibration_eye_point_left)
        x_l = np.linalg.lstsq(A_l, B, rcond=None)[0]
        computed_expected_l = list(zip(A_l.dot(x_l), B))
        print('right: ', computed_expected_l)

        #right eye
        A_r = np.vstack(calibration_eye_point_right)
        x_r = np.linalg.lstsq(A_r, B, rcond=None)[0]

        computed_expected_r = list(zip(A_r.dot(x_r), B))
        print('right: ', computed_expected_r)
        print(x_l)
        print(x_r)

        self.explanation.delete('1.0', tk.END)
        self.explanation.insert(tk.END, f"Parameters succesfully computed!\nNow you can start using the gaze mouse.")


    def close_window(self):
        calibration_done = True
        self.master.destroy()


class Precision:
    def __init__(self, master, number):
        self.master = master
        self.master.attributes('-fullscreen', True)
        self.frame = tk.Frame(self.master)
        self.explanation = tk.Text(self.frame, height="2", font="Helvetica")
        self.explanation.pack()
        self.explanation.insert(
            tk.END, "You have to click in order on the red dot in sequence and wait!")
        self.back = tk.Button(
            self.frame, text=f"<- Quit Precision Test!", fg="red", command=self.close_window)
        self.back.pack()
        self.calibration_points()
        self.frame.pack()

    def create_circle(self, x, y, canvas):  # center coordinates, radius
        r = 5
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        return canvas.create_oval(x0, y0, x1, y1, fill="red", activefill='orange')

    def drawCircle(self, x, y, row, column):
        self.circular_button = self.create_circle(x, y, self.canvas)
        button_number = column+(row-1)*3
        self.canvas.create_text(x, y-15, text=f"{button_number}")
        self.canvas.tag_bind(self.circular_button, "<Button-1>", lambda event,
                             circle_button=self.circular_button: self.compute_precision(event, button_number))

    def calibration_points(self):
        self.canvas = tk.Canvas(self.master)
        user32 = ctypes.windll.user32
        padding = 100
        x = (user32.GetSystemMetrics(0)-2*padding)/2
        y = (user32.GetSystemMetrics(1)-2*padding)/2
        for row in range(3):
            for column in range(3):
                self.drawCircle(x*column+padding, y*row + padding, row+1, column+1)
                point = (x*column+padding, y*row+padding)
                calibration_point.append(point)
        self.canvas.pack(fill=tk.BOTH, expand=1)

    def compute_precision(self, event, id_button):
        self.explanation.delete('1.0', tk.END)
        self.explanation.insert(tk.END, f"You have pressed {id_button}th point!\n Precision Test started!")
        X_l = build_unknown_array(x_l)
        X_r = build_unknown_array(x_r)
        cap = cv2.VideoCapture(settings.camera)

        n = 60
        i = 0
        avg_error_l = 0
        avg_error_r = 0
        avg_error = 0
        while(i < n):  # average out 5 consequents values
            _, frame = cap.read()
            iris_info = ir_pos.irides_position_form_video(frame)
            try:
                _, _, rel_l, rel_r = next(iris_info)
                if(rel_l is None or rel_r is None):
                    continue
                iris_l_param = build_iris_param_array(rel_l)
                iris_r_param = build_iris_param_array(rel_r)
                #print(iris_l_param)
                gaze_l = X_l.dot(iris_l_param)
                gaze_r = X_r.dot(iris_r_param)
                gaze = abs(((gaze_l+gaze_r)/2))
                #TODO: here i compute the error
                error_l = abs(gaze_l-calibration_point[id_button-1])
                error_r = abs(gaze_r-calibration_point[id_button-1])
                error = abs(gaze-calibration_point[id_button-1])
                avg_error_l = (avg_error_l+error_l)/2
                avg_error_r = (avg_error_r+error_l)/2
                avg_error = (avg_error+error_l)/2
                i += 1
            except StopIteration:
                pass
        print('-------------------------------'+ str(id_button) +'------------------------------------')
        print('Expected value: ' +str(calibration_point[id_button-1]))
        print('Computed gaze: ' + str(gaze))
        print('Avg_error_l, Avg_error_r, Avg_error')
        print(avg_error_l, avg_error_r, avg_error)
        cap.release()

    def close_window(self):
        calibration_done = True
        self.master.destroy()

if __name__ == "__main__":
    if debug:
        user32 = ctypes.windll.user32
        print("Resolution parameters. \nWidth =", user32.GetSystemMetrics(0)*2)
        print("Height =", user32.GetSystemMetrics(1)*2)
    root = tk.Tk()
    app = Home(root)
    root.mainloop()
