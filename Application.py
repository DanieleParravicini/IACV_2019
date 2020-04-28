import tkinter as tk
import time
import threading
import ctypes
import numpy as np
import iris_position as ir_pos
import cv2
import mouse

camera_number               = 1
calibration_done            = False
debug                       = False
calibration_eye_point_left  = []
calibration_eye_point_right = []
calibration_point           = []
x_l = [-32135.00893827, 100083.69427809,  85509.40189455, -198923.71419596,
       -21350.62179516, -3015.41128191, 131978.89576158, -3424.1461178,
       9605.44074001, -131035.97242032]  # calibration parameters for me to test overwritten by calibration
x_r = [-18417.29918659, 78282.52187778, -12735.70671031, 33758.59163978,
       -77485.32031864, 63486.07953807, 5302.47117536, -67208.48323979,
       -31413.2896205, 36351.51290157]

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
        self.frame.pack()

    def mouseControl(self):
        self.master.iconify()
        #Todo: here i compute and control the mouse position.
        X_l = np.array([[x_l[0], x_l[1], x_l[2], x_l[3], x_l[4], x_l[0]], [x_l[0], x_l[5], x_l[6], x_l[7], x_l[8], x_l[9]]])
        X_r = np.array([[x_r[0], x_r[1], x_r[2], x_r[3], x_r[4], x_r[0]], [x_r[0], x_r[5], x_r[6], x_r[7], x_r[8], x_r[9]]])
        cap = cv2.VideoCapture(camera_number)

        while True:
            _, frame = cap.read()
            iris_info = ir_pos.irides_position_form_video(frame)
            try:
                _, _, rel_l, rel_r = next(iris_info)
                if(rel_l is None or rel_r is None):
                    continue
                iris_l_param = np.array([1, rel_l[0], rel_l[1], rel_l[0]*rel_l[1], rel_l[0]**2, rel_l[1]**2])
                iris_r_param = np.array([1, rel_r[0], rel_r[1], rel_r[0]*rel_r[1], rel_r[0]**2, rel_r[1]**2])
                print(iris_l_param)
                gaze_l = X_l.dot(iris_l_param)
                gaze_r = X_r.dot(iris_r_param)
                gaze = abs(((gaze_l+gaze_r)/2))
                #print(gaze)
                mouse.move(gaze[0], gaze[1], absolute=True, duration=0)
            except StopIteration:
                pass


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
        cap = cv2.VideoCapture(camera_number)
    
        n = 60
        v_l = 0
        v_r = 0
        i = 0
        while(i < n):                                              #average out 5 consequents values
            _, frame   = cap.read()
            iris_info  = ir_pos.irides_position_form_video(frame)
            try:
                _, _, rel_l, rel_r = next(iris_info)
                if(rel_l is None or rel_r is None ):
                    continue

                v_l                 += rel_l
                v_r                 += rel_r
                i                   += 1
                
            except StopIteration:
                pass

        v_l = v_l/n
        v_r = v_r/n

        cap.release()

        calibration_eye_point_left.append(np.array([1+v_l[1]**2, v_l[0], v_l[1], v_l[0]*v_l[1], v_l[0]**2, 0, 0, 0, 0, 0]))
        calibration_eye_point_left.append(np.array([1, 0, 0, 0, 0, v_l[0], v_l[1], v_l[0]*v_l[1], v_l[0]**2, v_l[1]**2]))
        calibration_eye_point_right.append(np.array([1+v_r[1]**2, v_r[0], v_r[1], v_r[0]*v_r[1], v_r[0]**2, 0, 0, 0, 0, 0]))
        calibration_eye_point_right.append(np.array([1, 0, 0, 0, 0, v_r[0], v_r[1], v_r[0]*v_r[1], v_r[0]**2, v_r[1]**2]))

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
        #right eye
        A_r = np.vstack(calibration_eye_point_right)
        x_r = np.linalg.lstsq(A_r, B, rcond=None)[0]
        print('Calibration parameters for left eye: ' + str(x_l))
        print('Calibration parameters for right eye: ' + str(x_r))
        self.explanation.delete('1.0', tk.END)
        self.explanation.insert(tk.END, f"Parameters succesfully computed!\nNow you can start using the gaze mouse.")

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
