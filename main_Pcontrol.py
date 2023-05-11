from uart_cloud_platform import set_cloud_platform_degree
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
from net.inception import InceptionResNetV1
import gc
import time
from multiprocessing import Process, Manager
###---------------------servo-control----------------------######
def btm_servo_control(offset_x):
    '''
    底部舵机的比例控制
    这里舵机使用开环控制
    '''
    offset_dead_block = 0.1# 偏移量死区大小
    btm_kp = 20 # 控制舵机旋转的比例系数
    last_btm_degree = 90# 上一次底部舵机的角度
    
    # 设置最小阈值
    if abs(offset_x) < offset_dead_block:
       offset_x = 0

    # offset范围在-50到50左右
    delta_degree = offset_x * btm_kp
    # 计算得到新的底部舵机角度
    next_btm_degree = last_btm_degree + delta_degree
    # 添加边界检测
    if next_btm_degree < 0:
        next_btm_degree = 0
    elif next_btm_degree > 180:
        next_btm_degree = 180
    
    return int(next_btm_degree)

def top_servo_control(offset_y):
    '''
    顶部舵机的比例控制
    这里舵机使用开环控制
    '''
    offset_dead_block = 0.1
    top_kp = 20# 控制舵机旋转的比例系数
    last_top_degree = 90# 上一次顶部舵机的角度

    # 如果偏移量小于阈值就不相应
    if abs(offset_y) < offset_dead_block:
        offset_y = 0

    # offset_y *= -1
    # offset范围在-50到50左右
    delta_degree = offset_y * top_kp
    # 新的顶部舵机角度
    next_top_degree = last_top_degree + delta_degree
    # 添加边界检测
    if next_top_degree < 0:
        next_top_degree = 0
    elif next_top_degree > 180:
        next_top_degree = 180
    
    return int(next_top_degree)

##########-----------face-recognization---------------------####### 
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
                    max_face =  max(rectangles, key=lambda face: (face[3]-face[2])*(face[1]-face[0]))
                    # print(max_face)
                    face_x = float((left + right)/2.0)
                    face_y = float((top + bottom)/2.0)
                    # 人脸在画面中心X轴上的偏移量
                    offset_x = -float(face_x / width - 0.5) * 2
                    # 人脸在画面中心Y轴上的偏移量
                    offset_y = -float(face_y / height - 0.5) * 2
                    return (offset_x,offset_y)
                    break
            
# 向共享缓冲栈中写入数据:
def write(stack, cam, top: int) -> None:
    print('Process to write: %s' % os.getpid())
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
        if len(stack) >= 15:
            frame = stack.pop(0)
    
            img_height, img_width,_ = frame.shape
            face_offset = init.recognize(frame) 
            print(face_offset)
            if face_offset is not None:
                # 计算下一步舵机要转的角度
                offset_x = face_offset[0]
                offset_y = face_offset[1]
                next_btm_degree = btm_servo_control(offset_x)
                next_top_degree = top_servo_control(offset_y)
                # 舵机转动
                set_cloud_platform_degree(next_btm_degree, next_top_degree)
                # 更新角度值
                last_btm_degree = next_btm_degree
                last_top_degree = next_top_degree
                print("X轴偏移量：{} Y轴偏移量：{}".format(offset_x, offset_y))
                print('底部角度： {} 顶部角度：{}'.format( next_top_degree, next_btm_degree))
            
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
    out.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    # 父进程创建缓冲栈，并传给各个子进程：
    q = Manager().list()
    pw = Process(target=write, args=(q, "http://admin:admin@192.168.1.2:8081", 20)) #
    pr = Process(target=read, args=(q,))
    pw.start()
    pr.start()
 
    pw.join()
    pr.join()
 
    pw.terminate()


