from tkinter import *
from PIL import Image,ImageTk

root_window =Tk()
root_window.title("人脸识别及跟随系统")
root_window.geometry("4096x2304")
def cap():
    pass

def OpenCam():
    pass
def recognize():

    Name_window = Toplevel()
    Name_window.title("Input name")
    root_window.geometry("400x400")
    # Name_window.geometry("+100+100")

    faceName_var = StringVar()
    faceName_entry = Entry(Name_window, width = 10, textvariable = faceName_var)
    faceName_entry.pack()
    faceName_entry.insert(0, "Input name")

    BtmImage = Image.open('gui/camera.png').resize((100,100))
    img = ImageTk.PhotoImage(image = BtmImage)
    capture_button = Button(Name_window,image = img ,command = cap, cursor = 'hand2')
    capture_button.pack()
    Name_window.config(Button = capture_button)


panel = Label(root_window)
frame = Image.open('/home/daniel/catkin_ws/src/keras-face-recognition/face_dataset/obama.jpg').resize((1920,1080))
imgtk = ImageTk.PhotoImage(image = frame)
# panel.imgtk = imgtk
panel.config(image = imgtk)
panel.place(x = 0, y = 0, anchor='nw')

MenuBar = Menu(root_window)

Option  =  Menu(MenuBar , tearoff = 0)
MenuBar.add_cascade(label = 'Option',menu = Option)
Option.add_command(label = 'Open Camera',command = OpenCam)
Option.add_command(label = 'Recognize',command = recognize)
Option.add_separator()
Option.add_command(label = 'Exit',command = root_window.quit)

Data = Menu(MenuBar , tearoff = 0 )
MenuBar.add_cascade(label = 'Data',menu =Data)
Data.add_command(label = 'Setup',command = recognize)
Data.add_command(label = 'Reload',command = recognize)




root_window.config(menu = MenuBar)
root_window.mainloop()
