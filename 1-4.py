from __future__ import print_function
from __future__ import division
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()
from PIL import Image
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QDialog, QGridLayout, QLabel, QPushButton
from PyQt5 import QtCore, QtGui
from matplotlib.image import imread
import math

class MyDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(850, 480)
        self.label = QLabel()
        layout = QGridLayout(self)
        layout.addWidget(self.label, 0, 0, 4, 4)

        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(11)


        self.loadImage = QPushButton('1.1 Load Image', self)
        self.loadImage.setGeometry(QtCore.QRect(20, 130, 201, 32))
        self.loadImage.setFont(font)
        
        self.loadImage.clicked.connect(self.showImage2)
        ##########
    

        self.colorSep = QPushButton('1.2 Color Separation', self)
        self.colorSep.setGeometry(QtCore.QRect(20, 210, 201, 32))
        self.colorSep.setFont(font)

        self.colorSep.clicked.connect(self.split_RGBThreeChannel)
        ##########
        

        self.colorTran = QPushButton('1.3 Color Transformation', self)
        self.colorTran.setGeometry(QtCore.QRect(20, 290, 201, 32))
        self.colorTran.setFont(font)

        self.colorTran.clicked.connect(self.color_trans)
        ##########
 

        self.Blending = QPushButton('1.4 Blending', self)
        self.Blending.setGeometry(QtCore.QRect(20, 370, 201, 32))
        self.Blending.setFont(font)

        self.Blending.clicked.connect(self.blen)
        ##########
         

        self.gaussianB2 = QPushButton('2.1 Gaussian Blur', self)
        self.gaussianB2.setGeometry(QtCore.QRect(260, 130, 201, 32))
        self.gaussianB2.setFont(font)

        self.gaussianB2.clicked.connect(self.GaussianBlur2)
        ##########
        

        self.bilateral = QPushButton('2.2 Bilatral filter', self)
        self.bilateral.setGeometry(QtCore.QRect(260, 210, 201, 32))
        self.bilateral.setFont(font)

        self.bilateral.clicked.connect(self.bilateralFi)
        ##########

        self.median = QPushButton('2.3 Median filter', self)
        self.median.setGeometry(QtCore.QRect(260, 290, 201, 32))
        self.median.setFont(font)

        self.median.clicked.connect(self.medianFi)
        ##########
        

        self.gaussianB3 = QPushButton('3.1 Gaussian Blur', self)
        self.gaussianB3.setGeometry(QtCore.QRect(500, 130, 201, 32))
        self.gaussianB3.setFont(font)

        self.gaussianB3.clicked.connect(self.gaussianblr3)
        ##########
        

        self.sobelX = QPushButton('3.2 Sobel X', self)
        self.sobelX.setGeometry(QtCore.QRect(500, 210, 201, 32))
        self.sobelX.setFont(font)

        self.sobelX.clicked.connect(self.sobelx)
        ##########
        

        self.sobelY = QPushButton('3.3 Sobel Y', self)
        self.sobelY.setGeometry(QtCore.QRect(500, 290, 201, 32))
        self.sobelY.setFont(font)

        self.sobelY.clicked.connect(self.sobel_y)
        ##########

        self.magni = QPushButton('3.4 Magnitude', self)
        self.magni.setGeometry(QtCore.QRect(500, 370, 201, 32))
        self.magni.setFont(font)

        self.magni.clicked.connect(self.magi)
        ##############
        

        self.resize = QPushButton('4.1 Resize', self)
        self.resize.setGeometry(QtCore.QRect(750, 130, 201, 32))
        self.resize.setFont(font)

        self.resize.clicked.connect(self.resize_fun)
        ##########
        

        self.translation = QPushButton('4.2 Translation', self)
        self.translation.setGeometry(QtCore.QRect(750, 210, 201, 32))
        self.translation.setFont(font)

        self.translation.clicked.connect(self.trans)
        ##########
            

        self.rotation = QPushButton('4.3 Rotation, Scaling', self)
        self.rotation.setGeometry(QtCore.QRect(750, 290, 201, 32))
        self.rotation.setFont(font)

        self.rotation.clicked.connect(self.angle)
        ##########
            

        self.shearing = QPushButton('4.4 Shearing', self)
        self.shearing.setGeometry(QtCore.QRect(750, 370, 201, 32))
        self.shearing.setFont(font)

        self.shearing.clicked.connect(self.shear)
        ##########

    def showImage2(self):

        img_address = 'C:/Users/s1551/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg'
        
        img = cv2.imread(img_address)
        cv2.imshow('My Image', img)
        size = img.shape
        
        print("Height: ",size[0])
        print("Width: ",size[1])

        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def split_RGBThreeChannel(self):
        img_address = 'C:/Users/s1551/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg'
        
        img = cv2.imread(img_address)

        B,G,R = cv2.split(img)
        zeros = np.zeros(img.shape[:2],dtype="uint8")
        cv2.imshow("B Channel",cv2.merge([B,zeros,zeros]))
        cv2.imshow("G Channel",cv2.merge([zeros,G,zeros]))
        cv2.imshow("R Channel",cv2.merge([zeros,zeros,R]))

        cv2.waitKey(0)
           
    def color_trans(self):
        img_address = 'C:/Users/s1551/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg'
        
        img = cv2.imread(img_address)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('OpenCV_fun', gray_img)

        B,G,R = cv2.split(img)

        
        cv2.imshow("Average weighted",cv2.merge([B,G,R]))
    
        cv2.waitKey(0)
    

    def blen(self):
        
        img_address1 = 'C:/Users/s1551/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Strong.jpg'
        img_address2 = 'C:/Users/s1551/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Weak.jpg'

        src1 = cv2.imread(img_address2,1)
        src2 = cv2.imread(img_address1,1)

        cv2.namedWindow("Blend")

        def on_trackbar(val):
            alpha = val / 255
            beta = ( 1.0 - alpha )
            dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
            cv2.imshow('Blend', dst)

        cv2.createTrackbar("Blend", "Blend" , 0, 255, on_trackbar)
        on_trackbar(0)

        cv2.waitKey()

    def GaussianBlur2(self):
        img_address = 'C:/Users/s1551/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_whiteNoise.jpg'
        
        img = cv2.imread(img_address,1)

        kernel_size = 5
        img_gb = cv2.GaussianBlur(img,(kernel_size, kernel_size), 0)
        cv2.imshow('GaussianBlur', img_gb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def bilateralFi(self):
        img_address = 'C:/Users/s1551/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_whiteNoise.jpg'
        
        img = cv2.imread(img_address,1)
 
        # Apply bilateral filter with d = 15
        # sigmaColor = sigmaSpace = 90
        bilateral = cv2.bilateralFilter(img, 15, 90, 90)
        cv2.imshow('Bilateral filter', bilateral)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def medianFi(self):
        img_address = 'C:/Users/s1551/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_pepperSalt.jpg'
        
        img = cv2.imread(img_address,1)
        

        m3 = cv2.medianBlur(img, 3)
        m5 = cv2.medianBlur(img, 5)

        cv2.imshow('Median Filter 3x3', m3)
        cv2.imshow('Median Filter 5x5', m5)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def gaussianblr3(self):
        img_address = 'C:/Users/s1551/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg'
        img = cv2.imread(img_address,1)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale', gray_img)

        #3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        power = np.exp(-(x**2+y**2))

        #Normalization
        power = power / power.sum()
        

        gau = signal.convolve2d(gray_img, power, boundary='symm', mode='same')

        plt.imshow(gau,cmap="gray")
        # plt.colorbar()
        plt.axis('off')
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    

    def sobelx(self):
        img_address = 'C:/Users/s1551/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg'
        img = cv2.imread(img_address,1)
        g = np.asarray([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y = np.mgrid[-1:2, -1:2]
        power = np.exp(-(x**2+y**2))

        power = power / power.sum()
                

        gau = signal.convolve2d(gray_img, power, boundary='symm', mode='same')

        img_convolved = signal.convolve2d(gau, g)
        abs_sobelx = np.absolute(img_convolved)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        thresh_min = 20
        thresh_max = 100
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        plt.imshow(sxbinary, cmap='gray')

        plt.axis('off')
        plt.show()

    def sobel_y(self):
        img_address = 'C:/Users/s1551/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg'
        img = cv2.imread(img_address,1)
        g1 = np.asarray([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y = np.mgrid[-1:2, -1:2]
        power = np.exp(-(x**2+y**2))

        power = power / power.sum()
                

        gau = signal.convolve2d(gray_img, power, boundary='symm', mode='same')

        img_convolvedy = signal.convolve2d(gau, g1)
        abs_sobely = np.absolute(img_convolvedy)
        scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))
        thresh_min = 20
        thresh_max = 100
        sybinary = np.zeros_like(scaled_sobel)
        sybinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        plt.imshow(sybinary, cmap='gray')

        plt.figure()
        plt.imshow(gau,cmap="gray")
        plt.axis('off')
        plt.show()



    def magi(self):

        img_address = 'C:/Users/s1551/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg'
        img = cv2.imread(img_address,1)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        x, y = np.mgrid[-1:2, -1:2]
        power = np.exp(-(x**2+y**2))
        power = power / power.sum()
        gau = signal.convolve2d(gray_img, power, boundary='symm', mode='same')

        g_x = np.asarray([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

        g_y = np.asarray([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])


        img_x = signal.convolve2d(gau, g_x)
        img_y = signal.convolve2d(gau, g_y)

        img_x2 = np.power(img_x,2)
        img_y2 = np.power(img_y,2)

        img_add = img_x2+img_y2

        img_final = np.sqrt(img_add)

        abs_sobel = np.absolute(img_final)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        thresh_min = 20
        thresh_max = 120
        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        plt.imshow(binary, cmap='gray')

        
        plt.axis('off')
        plt.show()

        print(scaled_sobel)

    def resize_fun(self):
        img_address = 'C:/Users/s1551/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png'
        img = cv2.imread(img_address,1)

        img_re = cv2.resize(img,(256,256))
        size = img_re.shape
        print("Height: ",size[0])
        print("Width: ",size[1])
       
        cv2.imshow('resize',img_re)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def trans(self):
        img_address = 'C:/Users/s1551/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png'
        img = cv2.imread(img_address,1)

        img_re = cv2.resize(img,(256,256))

        M = np.float32([[1, 0, 0], [0, 1, 60]])

        dst = cv2.warpAffine(img_re, M, (400,300))
        cv2.imshow('translation', dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def angle(self):
        img_address = 'C:/Users/s1551/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png'
        img = cv2.imread(img_address,1)

        img_re = cv2.resize(img,(256,256))

        M = np.float32([[1, 0, 0], [0, 1, 60]])

        dst = cv2.warpAffine(img_re, M, (400,300))
        rows,cols = dst.shape[:2]

        Mr = cv2.getRotationMatrix2D((cols/2,rows/2),10,0.5)

        img2 = cv2.warpAffine(dst,Mr,(400,300))

        cv2.imshow('translation', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def shear(self):
        img_address = 'C:/Users/s1551/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png'
        img = cv2.imread(img_address,1)

        img_re = cv2.resize(img,(256,256))

        M = np.float32([[1, 0, 0], [0, 1, 60]])

        dst = cv2.warpAffine(img_re, M, (400,300))
        rows,cols = dst.shape[:2]

        Mr = cv2.getRotationMatrix2D((cols/2,rows/2),10,0.5)

        img2 = cv2.warpAffine(dst,Mr,(400,300))

        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[10,100],[200,50],[100,250]])
        M2 = cv2.getAffineTransform(pts1,pts2)

        img3 = cv2.warpAffine(img2,M2,(400,300))
        cv2.imshow('shearing', img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    a = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    sys.exit(a.exec_())