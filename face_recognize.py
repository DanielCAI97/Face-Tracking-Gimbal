from uart_cloud_platform import set_cloud_platform_degree
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
from net.inception import InceptionResNetV1
from tkinter import *  #设计前面板
from tkinter import messagebox
from PIL import Image,ImageTk
import math
import  xml.dom.minidom
from xml.dom import Node
from multiprocessing import Process, Manager
import threading
import gc
import time

def open_Cam():
    # global video_capture
    success, img = video_capture.read()  # 从摄像头读取照片
    if success:
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
        return cv2image

def image_show():
    global camOpen_running
    global faceRecog_running
    global cam
    camOpen_running = TRUE
    faceRecog_running = FALSE
    print("camera:%d, facerec:%d" %(camOpen_running,faceRecog_running))
    while (faceRecog_running == FALSE and camOpen_running == TRUE):
                cv2image = open_Cam()
                current_image = Image.fromarray(cv2image).resize((1920,1080))  # 将图像转换成Image对象
                imgtk = ImageTk.PhotoImage(image=current_image)
                # panel.imgtk = imgtk
                panel.config(image=imgtk)
                panel.update()

def exit():
    global camOpen_running
    global faceRecog_running
    # global video_capture
    camOpen_running = FALSE
    faceRecog_running = FALSE
    video_capture.release()
    root_window.quit()

class Params():
    face_3d_points = np.array(([-32, 35, 40],  # left eye
                               [31, 35, 41],  # right eye
                               [0, 0, 0],  # nose
                               [-25, -36, 17],  # left lip
                               [22, -36, 15]), dtype=np.float)  # right lip

    dom = xml.dom.minidom.parse('/home/daniel/catkin_ws/src/keras-face-recognition/cameraParam/XiaoMiCameraParam.xml')
    root = dom.documentElement
    # 相机参数
    opencv_matrix = dom.getElementsByTagName('data')[0].childNodes[0].nodeValue
    opencv_matrix = opencv_matrix.split()
    opencv_matrix = np.ascontiguousarray(opencv_matrix, dtype=np.float32)
    camera_matrix = np.reshape(opencv_matrix, (3, 3))
    #相机畸变
    distortion_coefficients = dom.getElementsByTagName('data')[1].childNodes[0].nodeValue
    distortion_coefficients = distortion_coefficients.split()
    distortion_coefficients = np.ascontiguousarray(distortion_coefficients, dtype=np.float32)

class face_rec():
    def __init__(self):
        global count
        count += 1
        if count > 1:
            messagebox.showinfo(title='Waiting ...', message='Please wait for the loading finished')
            # 创建mtcnn对象
            # 检测图片中的人脸
            self.mtcnn_model = mtcnn()
            # 门限函数
            self.threshold = [0.5,0.8,0.9]

            # 载入facenet
            # 将检测到的人脸转化为128维的向量
            self.facenet_model = InceptionResNetV1()
            # model.summary()
            model_path = './model_data/facenet_keras.h5'
            self.facenet_model.load_weights(model_path)

            #-----------------------------------------------#
            #   对数据库中的人脸进行编码
            #   known_face_encodings中存储的是编码后的人脸
            #   known_face_names为人脸的名字
            #-----------------------------------------------#
            face_list = os.listdir("face_dataset")

            self.known_face_encodings=[]

            self.known_face_names=[]

            for face in face_list:
                name = face.split(".")[0]

                img = cv2.imread("./face_dataset/"+face)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

                # 检测人脸
                rectangles = self.mtcnn_model.detectFace(img, self.threshold)

                # 转化成正方形
                rectangles = utils.rect2square(np.array(rectangles))
                # facenet要传入一个160x160的图片
                rectangle = rectangles[0]
                # 记下他们的landmark
                landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160

                crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
                crop_img = cv2.resize(crop_img,(160,160))

                new_img,_ = utils.Alignment_1(crop_img,landmark)

                new_img = np.expand_dims(new_img,0)
                # 将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
                face_encoding = utils.calc_128_vec(self.facenet_model,new_img)

                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)

            messagebox.showinfo(title = 'Done',message = 'Finshed loading')

    def image_read(self, stack, top: int) -> None:
        global faceRecog_running,camOpen_running
        faceRecog_running = TRUE
        while (faceRecog_running == TRUE and camOpen_running == TRUE):
            success, img = video_capture.read()  # 从摄像头读取照片
            if success:
                stack.append(img)
                # 每采集8帧清空缓冲
                if len(stack) >= top:
                    del stack[:]
                    gc.collect()

    def image_process(self, stack) -> None:
        #-----------------------------------------------#
        #   人脸识别
        #   先定位，再进行数据库匹配
        #-----------------------------------------------#
        global faceRecog_running, camOpen_running
        while (faceRecog_running == TRUE and camOpen_running == TRUE):
            #当缓冲区数据大小超过8时，读取最近的一次数据
            if len(stack) >= 8:
                draw = stack.pop(0)
                draw_rgb = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)
                process_lock.acquire()
                height,width,_ = np.shape(draw_rgb)
                # 检测人脸
                rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)
                if len(rectangles)!=0:
                    # 转化成正方形
                    rectangles = utils.rect2square(np.array(rectangles,dtype=np.int32))
                    rectangles[:,0] = np.clip(rectangles[:,0],0,width)
                    rectangles[:,1] = np.clip(rectangles[:,1],0,height)
                    rectangles[:,2] = np.clip(rectangles[:,2],0,width)
                    rectangles[:,3] = np.clip(rectangles[:,3],0,height)
                    #-----------------------------------------------#
                    #   对检测到的人脸进行编码
                    #-----------------------------------------------#
                    face_encodings = []
                    for rectangle in rectangles:
                        landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160

                        crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
                        crop_img = cv2.resize(crop_img,(160,160))

                        new_img,_ = utils.Alignment_1(crop_img,landmark)
                        new_img = np.expand_dims(new_img,0)

                        face_encoding = utils.calc_128_vec(self.facenet_model,new_img)
                        face_encodings.append(face_encoding)

                    face_names = []
                    for face_encoding in face_encodings:
                        # 取出一张脸并与数据库中所有的人脸进行对比，计算得分
                        matches = utils.compare_faces(self.known_face_encodings, face_encoding, tolerance = 0.9)
                        name = "Unknown"
                        # 找出距离最近的人脸
                        face_distances = utils.face_distance(self.known_face_encodings, face_encoding)
                        # 取出这个最近人脸的评分
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                        face_names.append(name)

                    rectangles = rectangles[:,0:4]
                    #-----------------------------------------------#
                    #   画框~!~
                    #-----------------------------------------------#

                    for (left, top, right, bottom), name in zip(rectangles, face_names):
                        cv2.rectangle(draw_rgb, (left, top), (right, bottom), (0, 0, 255), 2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(draw_rgb, name, (left , bottom - 15), font, 0.75, (255, 255, 255), 2)
                        
                        global tracing_running
                        if tracing_running == TRUE:
                            if name == "cai":
                                # 求解相机位姿
                                face_2d_points = np.reshape(rectangle[5:15], (5, 2))
                                face_2d_points = np.ascontiguousarray(face_2d_points, dtype=np.float32)
                                retval, rvec, tvec = cv2.solvePnP(Params.face_3d_points, face_2d_points,
                                                                  Params.camera_matrix, Params.distortion_coefficients,
                                                                  flags=cv2.SOLVEPNP_EPNP)
                                print(tvec)
                                nose_x = rectangle[9]
                                nose_y = rectangle[10]
                                # rotM = cv2.Rodrigues(rvec)[0]
                                # camera_postion = -np.matrix(rotM).T * np.matrix(tvec)
                                x = tvec[0]
                                y = tvec[1]
                                r = tvec[2]
                                x = -x
                                y = -y

                                print('x:%d ,y:%d ,r:%d' % (x, y, r))
                                global last_x
                                global last_y
                                TM = math.atan(y / r) * 180 / math.pi
                                BM = math.atan(x / r) * 180 / math.pi
                                if (abs(TM) < 3):
                                    TM = 0
                                else:
                                    TM = TM
                                if (abs(BM) < 3):
                                    BM = 0
                                else:
                                    BM = BM
                                print("delta BOTTOM:%d , delta TOP:%d" % (BM, TM))
                                global last_TM, last_BM
                                next_TM = last_TM + int(TM)
                                next_BM = last_BM + int(BM)
                                # print("roll:%d, pitch:%d" %(next_roll,next_pitch))
                                # 舵机转动
                                if next_TM >= 100:
                                    next_TM = 100
                                elif next_TM < 80:
                                    next_TM = 80
                                if next_BM >= 180:
                                    next_BM = 180
                                elif next_BM < 0:
                                    next_BM = 0
                                print("bottom:%d , top:%d" % (next_BM, next_TM))
                                set_cloud_platform_degree(next_BM, next_TM)
                                last_BM = next_BM
                                last_TM = next_TM

                current_image = Image.fromarray(draw_rgb).resize((1920, 1080))  # 将图像转换成Image对象
                imgtk = ImageTk.PhotoImage(image=current_image)
                panel.config(image=imgtk)
                panel.update()
                print("release2")
                process_lock.release()


    def recognize(self):
        global camOpen_running
        global  faceRecog_running
        faceRecog_running = TRUE
        image_read = threading.Thread(target = self.image_read, args = (stack, 8,))
        image_read.start()
        image_process = threading.Thread(target = self.image_process, args = (stack, ))
        image_process.start()
    
    def tracing(self):
        global tracing_running
        tracing_running = TRUE




class Face_Data():
#-----------------------------------------------#
    #   人脸抓取
#-----------------------------------------------#
    def FaceSeize(self):
        global camOpen_running
        global faceRecog_running
        global Name_window
        faceRecog_running = FALSE
        if (faceRecog_running == FALSE and camOpen_running == TRUE):
            face_name = faceName_var.get()
            if face_name != "Input name":
                success, img = video_capture.read()  # 从摄像头读取照片
                if success:
                    # cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
                    imgname = "/home/daniel/catkin_ws/src/keras-face-recognition/face_dataset/" + face_name + ".jpg";
                    cv2.imwrite(imgname, img)
                    Name_window.quit()
                    messagebox.showinfo(title='Finshed capture', message = face_name+'is added to Repository')
            if face_name == "Input name":
                messagebox.showwarning(title='Retry', message = 'Please rewrite the true name!!')
        elif (faceRecog_running == FALSE and camOpen_running == FALSE):
            messagebox.showerror(title='Error', message='Please open the camera!!')



    def Create_Window(self):
        global  Name_window
        Name_window = Toplevel()
        Name_window.title("Input name")
        root_window.geometry("400x400")
        # Name_window.geometry("+100+100")
        global faceName_var
        faceName_var = StringVar()
        faceName_entry = Entry(Name_window, width = 10, textvariable = faceName_var)
        faceName_entry.pack()
        faceName_entry.insert(0, "Input name")

        BtmImage = Image.open('gui/camera.png').resize((100,100))
        img = ImageTk.PhotoImage(image = BtmImage)
        capture_button = Button(Name_window,image = img ,command = self.FaceSeize, cursor = 'hand2')
        capture_button.pack()
        Name_window.config(Button = capture_button)



class FrontBoard():
    def __init__(self):
        global root_window
        root_window = Tk()
        root_window.title("人脸识别及跟随系统")
        root_window.geometry("4096x2304")

        global panel
        panel = Label(root_window)
        panel.place(x=0, y=0, anchor='nw')

        MenuBar = Menu(root_window)

        global Option
        Option = Menu(MenuBar, tearoff=0)
        MenuBar.add_cascade(label='Option', menu=Option)
        Option.add_command(label='Open Camera', command = image_show)
        Option.add_command(label='Recognize', command = dududu.recognize)
        Option.add_command(label='Tracing', command = dududu.tracing)
        Option.add_separator()
        Option.add_command(label='Exit', command = exit)

        Data = Menu(MenuBar, tearoff=0)
        MenuBar.add_cascade(label='Data', menu=Data)
        Data.add_command(label='Setup', command = data.Create_Window)
        Data.add_command(label='Reload', command= dududu.__init__)
        root_window.config(menu=MenuBar)
        root_window.mainloop()

if __name__ == "__main__":
    count = 0
    stack = Manager().list()
    process_lock = threading.Lock()
    camOpen_running = FALSE
    faceRecog_running = FALSE
    tracing_running = FALSE
    last_TM = 90
    last_BM = 90
    cam = input("输入1为电脑自带摄像头  \n输入2为手机摄像头 \n请选择摄像头：")
    if cam == '1':
        cam = 0
    if cam == '2':
        cam = "http://admin:admin@192.168.1.2:8081"
    video_capture = cv2.VideoCapture(cam)
    # 设置缓存区的大小 !!!
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    data = Face_Data()
    dududu = face_rec()
    Init_window = FrontBoard()







