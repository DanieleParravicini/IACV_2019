import tkinter as tk
import time
import ctypes
import numpy as np
import iris_position as ir_pos
import cv2

camera_number               = 1
calibration_done            = False
debug                       = False


class Home:
    def __init__(self, master):
        self.master = master
        self.master.geometry("400x400")
        self.frame = tk.Frame(self.master)
        if(calibration_done == False):
            tk.Label(self.frame, text="Get images to compute precision:").pack(
                side="top")
        self.butnew("Get Test Images", "2", ImageGet)
        self.frame.pack()

    def butnew(self, text, number, _class):
        tk.Button(self.frame, text=text,
                  command=lambda: self.new_window(number, _class)).pack()

    def new_window(self, number, _class):
        self.new = tk.Toplevel(self.master)
        _class(self.new, number)

class ImageGet:
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
        button_number = column+(row-1)*5
        self.canvas.create_text(x,y-15, text=f"{button_number}")
        # self.clicked just for testing but than it will call calibrate
        self.canvas.tag_bind(self.circular_button, "<Button-1>", lambda event, circle_button = self.circular_button: self.getImage(event, button_number))

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
        x = (user32.GetSystemMetrics(0)-2*padding)/4
        y = (user32.GetSystemMetrics(1)-2*padding)/4
        for row in range(5):
            for column in range(5):
                self.drawCircle(x*column+padding, y*row+padding, row+1, column+1)
        self.canvas.pack(fill=tk.BOTH, expand=1)

    def getImage(self, event, id_button):
        self.explanation.delete('1.0', tk.END)
        self.explanation.insert(tk.END, f"You have pressed {id_button}th point!\n Calibration started!")
        cap = cv2.VideoCapture(camera_number)
        _, frame = cap.read()
        cv2.imwrite("iris_position_test_directory_gio/%d_image.jpg" %
                    id_button, frame)
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
