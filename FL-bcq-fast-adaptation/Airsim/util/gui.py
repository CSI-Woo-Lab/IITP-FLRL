from tkinter import *
from PIL import ImageTk, Image
import threading
import numpy as np


class gui(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self, name=name)
        self.start()

    def changeImage(self, newImage):
        self.label.configure(image=newImage)
        self.label.image = newImage

    def run(self):
        self.root = Tk()
        self.root.title("자율주행 Agent가 보는 이미지")
        self.root.geometry('480x240+50+50')
        image = Image.fromarray(np.uint8(np.zeros([80,160,3])))
        image = image.resize((480, 240), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)
        self.label = Label(self.root, image=image)
        self.label.pack()

        self.root.mainloop()
