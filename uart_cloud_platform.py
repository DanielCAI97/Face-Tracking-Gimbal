# -*- coding:utf-8 -*-
'''
PC使用PySerial发送数据
负责与ESP32主控的舵机云台进行通信
------------------------------------
## 注意
运行此程序的时候，需要修改设备的权限，
sudo chmod 777 /dev/ttyUSB?  ，其中 ? = 0,1,2,...
或者使用管理员权限运行脚本
sudo python xxxxx.py

'''
import serial
import struct
import time

import rospy
import roslib
from std_msgs.msg import String
# 串口号 默认为 /dev/ttyUSB0
ser_dev = '/dev/ttyUSB0c'
# 创建一个串口实例
ser = serial.Serial(ser_dev, 19200, timeout=1, bytesize=8)


# def pack_bin_data(bottom_degree, top_degree):
#     '''
#     h: unsigned short bit=2
#     b: unsigned char (byte): bit =1
#     '''

#     bin_data = struct.pack(">iiB",
#         int(bottom_degree), # 顶部舵机的角度
#         int(top_degree), # 底部舵机的角度
#         0x0A) # 结束符 '\n = 0x0A
#     return bin_data


def set_cloud_platform_degree(bottom_degree, top_degree):
    global ser
    byte_raw ='R'+' '+str('%02x' %top_degree)+'P'+' '+str('%02x' %bottom_degree)
    ser.write(byte_raw.encode("gbk"))
    # 讲接收的字节流转换成容易读取的样式
    # byte_str = ':'.join(["{:02x}".format(char_byte) for char_byte in bytearray(byte_raw)])
    # 打印原始的字节数据()

    # print("Send字节流: "+byte_str+"\n")
    return(byte_raw)

if __name__ == "__main__":


    while True:
        # 测试角度
        a = set_cloud_platform_degree(90, 90)
        print(a)


	    
        # 每隔10s发送一次数据
        time.sleep(1)
	
