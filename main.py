from uart_cloud_platform import set_cloud_platform_degree
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
import numpy as np
import math
from net.mtcnn import mtcnn
import utils.utils as utils
from net.inception import InceptionResNetV1
import gc
import time
from multiprocessing import Process, Manager
# from uart_cloud_platform import set_cloud_platform_degree
import math
import  xml.dom.minidom
from xml.dom import Node

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


#-----------------------------------------------#
    #   人脸识别的库
#-----------------------------------------------#
class face_rec():
    def __init__(self):
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


    def recognize(self,draw):
            #-----------------------------------------------#
            #   人脸识别
            #   先定位，再进行数据库匹配
            #-----------------------------------------------#
            height,width,_ = np.shape(draw)
            draw_rgb = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)

            # 检测人脸
            rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)

            if len(rectangles)==0:
                return

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
            face_2d_points = []
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
                cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)          

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(draw, name, (left , bottom - 15), font, 0.75, (255, 255, 255), 2) 
                
                if name == "cai":
                    # 求解相机位姿
                    face_2d_points = np.reshape(rectangle[5:15], (5, 2))
                    face_2d_points = np.ascontiguousarray(face_2d_points, dtype=np.float32)
                    retval, rvec, tvec = cv2.solvePnP(Params.face_3d_points, face_2d_points, Params.camera_matrix,Params.distortion_coefficients, flags=cv2.SOLVEPNP_EPNP)
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

                    # print('x:%d ,y:%d ,r:%d' % (x, y, r))
                    global last_x
                    global last_y
                    TM = math.atan(y / r) * 180 / math.pi
                    BM = math.atan(x / r) * 180 / math.pi
                    if(abs(TM) < 3):
                        TM = 0
                    else:
                        TM = TM
                    if(abs(BM) < 3):
                        BM = 0
                    else:
                        BM = BM
                    # print("delta BOTTOM:%d , delta TOP:%d" % (BM, TM))
                    return (BM,TM)


            
            
# 向共享缓冲栈中写入数据:
def write(stack, cam, top: int) -> None:
    print('Process to write: %s' % os.getpid())
    global cap
    cap = cv2.VideoCapture(cam)
    while True:
        _, img = cap.read()
        # 设置缓存区的大小 !!!
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        # img = cv2.flip(img, 1)
        if _:
            stack.append(img)
            # 每到一定容量清空一次缓冲栈
            # 利用gc库，手动清理内存垃圾，防止内存溢出
            if len(stack) >= top:
                del stack[:]
                gc.collect()
 
 
# 在缓冲栈中读取数据:
def read(stack) -> None: #提醒返回值是一个None
    print('Process to read: %s' % os.getpid())
    index = 0

 
    start_time = time.time()
    x = 1  # displays the frame rate every 1 second
    counter = 0
    
    init = face_rec()
    print("开始逐帧读取")
    while True:
        # print("正在读取第%d帧：" %index)
        if len(stack) >= 1:
            frame = stack.pop(0)
    
            img_height, img_width,_ = frame.shape
            angle = init.recognize(frame)
            if angle is not None:
                bottom_degree = angle[0]
                top_degree = angle[1]
                global last_TM,last_BM
                next_TM = last_TM + int(top_degree)
                next_BM = last_BM + int(bottom_degree)
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
                # print("bottom:%d , top:%d" % (next_BM, next_TM))
                set_cloud_platform_degree( next_BM, next_TM)
                last_BM = next_BM
                last_TM = next_TM

                index = index+1
           
            #直接保存视频
            cv2.imshow("FaceDetect", frame)
 
            #计算fps
            counter += 1
            if (time.time() - start_time) > x:
 
                print("FPS: ", counter / (time.time() - start_time))
                counter = 0
                start_time = time.time()
 
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            continue
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    # 父进程创建缓冲栈，并传给各个子进程：
    last_TM = 90
    last_BM = 90
    last_x = 0
    last_y = 0
    q = Manager().list()
    pw = Process(target=write, args=(q, "http://admin:admin@192.168.1.2:8081", 15)) #
    # pw = Process(target=write, args=(q, 0, 15)) #
    pr = Process(target=read, args=(q, ))
    pw.start()
    pr.start()


 
    pw.join()
    pr.join()
 
    pw.terminate()


