import tkinter as tk
import time
import threading
import ctypes
import numpy as np
import iris_position as ir_pos

calibration_done = False
debug = False
calibration_eye_point_left = []
calibration_eye_point_right = []
calibration_point = []

class Home:
    def __init__(self, master):
        self.master = master
        self.master.geometry("400x400")
        self.frame = tk.Frame(self.master)
        if(calibration_done == False):
            tk.Label(self.frame, text="Calibration required before use:").pack(
                side="top")
        self.butnew("Calibration", "2", Calibration)
        self.butnew("Gaze Draw", "3", Application)
        self.frame.pack()

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
        point_position = self.calibration_points()
        self.frame.pack()

    def create_circle(self, x, y, canvas):  # center coordinates, radius
        r = 5
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        return canvas.create_oval(x0, y0, x1, y1, fill="red")

    def drawCircle(self, x, y, row, column):
        self.circular_button = self.create_circle(x, y, self.canvas)
        self.test = self.circular_button
        button_number = column+(row-1)*3
        # self.clicked just for testing but than it will call calibrate
        self.canvas.tag_bind(self.circular_button, "<Button-1>", lambda event, circle_button = self.circular_button: self.clicked(event, button_number))
    
    def clicked(self, event, id_button):
        self.canvas.itemconfig(self.circular_button, fill="green")
        # to be investigate how to change color over circle presssed
        self.explanation.delete('1.0', tk.END)
        self.explanation.insert(tk.END, f"You have pressed {id_button}th!")
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
        return calibration_point
    
 
    def calibrate(self, x, y, row, column):
        v_l, v_r = ir_pos.irides_position_form_video(1)
        for i in range(5):                                      #average out 5 consequents values
            v_l, v_r = (ir_pos.irides_position_form_video(1))/2
        calibration_eye_point_left.append(v_l)
        calibration_eye_point_right.append(v_r)                 #Maybe the tuples should be converted in array
                                                                #consider to change the visual appearance of the dot to give a feedback  to user
    
    def compute_parameters(self):
        #devo risolvere un sistema di 9*2 equazioni in (6*2-3)*2 incognite
        #for each point i call a function that calls gaze position and returns the actual eye position and i average out this value.
        # A=[1+v_x^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #   [1+v_x^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #   [1+v_x^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #   [1+v_x^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #   [1+v_x^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #   [1+v_x^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #   [1+v_x^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #   [1+v_x^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #   [1+v_x^2, v_x, v_y, v_x*v_y, v_x^2, 0, 0, 0, 0]
        #   [1, 0, 0, 0, 0, v_x, v_y, v_x*v_y, v_x^2, v_y^2]
        #x=[a_0, a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4, b_5]
        #V=[s_x,s_y,s_x,s_y,s_x,s_y,s_x,s_y,s_x,s_y,s_x,s_y,s_x,s_y,s_x,s_y,s_x,s_y]'
        
        #left eye
        A_l = np.array(calibration_eye_point_left)          #not tested yet (maybe problems for array dimentions)
        x_l = np.linalg.solve(A_l,calibration_point)
        #right eye
        A_r = np.array()
        V = np.array(calibration_eye_point_right)
        x_r = np.linalg.solve(A_r, calibration_point)
    
    def hide_me(self, event):
        print('hide me')
        event.place_forget()

    def close_window(self):
        calibration_done = True
        self.master.destroy()


class Application:
    def __init__(self, master, number):
        self.master = master
        self.master.geometry("400x400+200+200")
        self.frame = tk.Frame(self.master)
        self.quit = tk.Button(
            self.frame, text=f"Quit this window n. {number}", command=self.close_window)
        self.quit.pack()
        self.label = tk.Label(
            self.frame, text="THIS IS ONLY IN THE THIRD WINDOW")
        self.label.pack()
        self.frame.pack()

    def close_window(self):
        self.master.destroy()


if __name__ == "__main__":
    if debug:
        user32 = ctypes.windll.user32
        print("Resolution parameters. \nWidth =", user32.GetSystemMetrics(0)*2)
        print("Height =", user32.GetSystemMetrics(1)*2)
    root = tk.Tk()
    app = Home(root)
    root.mainloop()
