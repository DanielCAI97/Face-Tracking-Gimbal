import  xml.dom.minidom
from xml.dom import Node
import numpy as np

#打开xml文档
dom = xml.dom.minidom.parse('XiaoMiFrontCameraParam.xml')
#得到文档元素对象
root = dom.documentElement

opencv_matrix = dom.getElementsByTagName('data')[0].childNodes[0].nodeValue
opencv_matrix = opencv_matrix.split()
opencv_matrix = np.ascontiguousarray(opencv_matrix , dtype = np.float32)
opencv_matrix = np.reshape(opencv_matrix,(3,3))
print(opencv_matrix)

distortion_coefficients = dom.getElementsByTagName('data')[1].childNodes[0].nodeValue
distortion_coefficients = distortion_coefficients.split()
distortion_coefficients = np.ascontiguousarray(distortion_coefficients , dtype = np.float32)






