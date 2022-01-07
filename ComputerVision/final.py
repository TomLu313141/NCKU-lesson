from __future__ import print_function
from __future__ import division

import seaborn as sns; sns.set()

import sys
from PyQt5.QtWidgets import QApplication, QDialog,QGridLayout, QLabel, QPushButton
from PyQt5 import QtCore, QtGui

from PyQt5.QtWidgets import (QApplication, QPushButton, QLabel, QInputDialog, QTextBrowser)
import cv2
import numpy as np
import glob


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


        self.loadImage = QPushButton('2.1', self)
        self.loadImage.setGeometry(QtCore.QRect(20, 130, 201, 32))
        self.loadImage.setFont(font)
        
        self.loadImage.clicked.connect(self.corn)
        # ##########
    

        self.colorSep = QPushButton('2.2', self)
        self.colorSep.setGeometry(QtCore.QRect(20, 210, 201, 32))
        self.colorSep.setFont(font)

        self.colorSep.clicked.connect(self.intr)
        # #########
 

        self.Blending = QPushButton('2.4', self)
        self.Blending.setGeometry(QtCore.QRect(20, 370, 201, 32))
        self.Blending.setFont(font)

        self.Blending.clicked.connect(self.dist)
        ##########

        self.Test = QPushButton('2.3', self)
        self.Test.setGeometry(QtCore.QRect(20, 290, 201, 32))
        self.Test.setFont(font)

        self.Test.clicked.connect(self.showDialog)

        ##########

        self.un = QPushButton('2.5', self)
        self.un.setGeometry(QtCore.QRect(20, 450, 201, 32))
        self.un.setFont(font)

        self.un.clicked.connect(self.undis)

    def undis(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值
        #棋盘格模板规格
        w = 11   # 12 - 1
        h = 8   # 9  - 1
        # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
        objp = np.zeros((w*h,3), np.float32)
        objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
        objp = objp*18.1  # 18.1 mm

        # 储存棋盘格角点的世界坐标和图像坐标对
        objpoints = [] # 在世界坐标系中的三维点
        imgpoints = [] # 在图像平面的二维点
        #加载pic文件夹下所有的jpg图像
        images = glob.glob('C:/Users/s1551/Desktop/HW2/Q2_Image/*.bmp')  #   拍摄的十几张棋盘图片所在目录

        j=0
        for fname in images:

            img = cv2.imread(fname)
            # 获取画面中心点
            #获取图像的长宽
            h1, w1 = img.shape[0], img.shape[1]
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            u, v = img.shape[:2]
            # 找到棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
            # 如果找到足够点对，将其存储起来
            if ret == True:
                print("j:", j)
                j = j+1
                # 在原角点的基础上寻找亚像素角点
                cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                #追加进入世界三维点和平面二维点中
                objpoints.append(objp)
                imgpoints.append(corners)


        ret, mtx, dist, rvecs, tvecs = \
            cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


        for fname in images:
            img = cv2.imread(fname)
            # if width:
            #     img = imutils.resize(img, width=width)

            h,  w = img.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

            # undistort
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

            # crop the image
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]

            dst2 = cv2.resize(dst,(512,512))
            img2 = cv2.resize(img,(512,512))

            cv2.imshow("undistorted", dst2)
            cv2.imshow("distorted", img2)

            
            cv2.waitKey(1000)


    def dist(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值
        #棋盘格模板规格
        w = 11   # 12 - 1
        h = 8   # 9  - 1
        # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
        objp = np.zeros((w*h,3), np.float32)
        objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
        objp = objp*18.1  # 18.1 mm

        # 储存棋盘格角点的世界坐标和图像坐标对
        objpoints = [] # 在世界坐标系中的三维点
        imgpoints = [] # 在图像平面的二维点
        #加载pic文件夹下所有的jpg图像
        images = glob.glob('C:/Users/s1551/Desktop/HW2/Q2_Image/*.bmp')  #   拍摄的十几张棋盘图片所在目录

        i=0
        for fname in images:

            img = cv2.imread(fname)
            # 获取画面中心点
            #获取图像的长宽
            h1, w1 = img.shape[0], img.shape[1]
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            u, v = img.shape[:2]
            # 找到棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
            # 如果找到足够点对，将其存储起来
            if ret == True:
                #print("i:", i)
                i = i+1
                # 在原角点的基础上寻找亚像素角点
                cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                #追加进入世界三维点和平面二维点中
                objpoints.append(objp)
                imgpoints.append(corners)


        ret, mtx, dist, rvecs, tvecs = \
            cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("Distortion:")
        print(dist)


    def intr(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值
        #棋盘格模板规格
        w = 11   # 12 - 1
        h = 8   # 9  - 1
        # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
        objp = np.zeros((w*h,3), np.float32)
        objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
        objp = objp*18.1  # 18.1 mm

        # 储存棋盘格角点的世界坐标和图像坐标对
        objpoints = [] # 在世界坐标系中的三维点
        imgpoints = [] # 在图像平面的二维点
        #加载pic文件夹下所有的jpg图像
        images = glob.glob('C:/Users/s1551/Desktop/HW2/Q2_Image/*.bmp')  #   拍摄的十几张棋盘图片所在目录

        i=0
        for fname in images:

            img = cv2.imread(fname)
            # 获取画面中心点
            #获取图像的长宽
            h1, w1 = img.shape[0], img.shape[1]
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            u, v = img.shape[:2]
            # 找到棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
            # 如果找到足够点对，将其存储起来
            if ret == True:
                # print("i:", i)
                i = i+1
                # 在原角点的基础上寻找亚像素角点
                cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                #追加进入世界三维点和平面二维点中
                objpoints.append(objp)
                imgpoints.append(corners)
                # 将角点在图像上显示
                cv2.drawChessboardCorners(img, (w,h), corners, ret)

        cv2.destroyAllWindows()

        #标定
        ret, mtx, dist, rvecs, tvecs = \
            cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("Intrinsic:\n",mtx)      # 内参数矩阵

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
        ##########
    def corn(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值
        #棋盘格模板规格
        w = 11   # 12 - 1
        h = 8   # 9  - 1
        # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
        objp = np.zeros((w*h,3), np.float32)
        objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
        objp = objp*18.1  # 18.1 mm

        # 储存棋盘格角点的世界坐标和图像坐标对
        objpoints = [] # 在世界坐标系中的三维点
        imgpoints = [] # 在图像平面的二维点
        #加载pic文件夹下所有的jpg图像
        images = glob.glob('C:/Users/s1551/Desktop/HW2/Q2_Image/*.bmp')  #   拍摄的十几张棋盘图片所在目录

        i=0
        for fname in images:

            img = cv2.imread(fname)
            # 获取画面中心点
            #获取图像的长宽
            h1, w1 = img.shape[0], img.shape[1]
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            u, v = img.shape[:2]
            # 找到棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
            # 如果找到足够点对，将其存储起来
            if ret == True:
                print("i:", i)
                i = i+1
                # 在原角点的基础上寻找亚像素角点
                cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                #追加进入世界三维点和平面二维点中
                objpoints.append(objp)
                imgpoints.append(corners)
                # 将角点在图像上显示
                cv2.drawChessboardCorners(img, (w,h), corners, ret)
                cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('findCorners', 640, 480)
                cv2.imshow('findCorners',img)
                cv2.waitKey(1000)
        cv2.destroyAllWindows()

        #标定
        ret, mtx, dist, rvecs, tvecs = \
            cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

   
    def showDialog(self):
        sender = self.sender()
      
        if sender == self.Test:
            text, ok = QInputDialog.getText(self, '編號', 'enter a number：')
            if ok:
                number = int(text)
                # print(number)

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值
                #棋盘格模板规格
                w = 11   # 12 - 1
                h = 8   # 9  - 1
                # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
                objp = np.zeros((w*h,3), np.float32)
                objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
                objp = objp*18.1  # 18.1 mm

                # 储存棋盘格角点的世界坐标和图像坐标对
                objpoints = [] # 在世界坐标系中的三维点
                imgpoints = [] # 在图像平面的二维点
                #加载pic文件夹下所有的jpg图像
                images = glob.glob('C:/Users/s1551/Desktop/HW2/Q2_Image/*.bmp')  #   拍摄的十几张棋盘图片所在目录

                i=0
                for fname in images:

                    img = cv2.imread(fname)
                    # 获取画面中心点
                    #获取图像的长宽
                    h1, w1 = img.shape[0], img.shape[1]
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    u, v = img.shape[:2]
                    # 找到棋盘格角点
                    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
                    # 如果找到足够点对，将其存储起来
                    if ret == True:
                        #print("i:", i)
                        i = i+1
                        # 在原角点的基础上寻找亚像素角点
                        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                        #追加进入世界三维点和平面二维点中
                        objpoints.append(objp)
                        imgpoints.append(corners)


                ret, mtx, dist, rvecs, tvecs = \
                    cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

          


                R = cv2.Rodrigues(rvecs[number])
                # print (R[0])
                # print(tvecs[0])

                print("Extrinsic:")
                print(np.concatenate([R[0], tvecs[number]], axis=1))

                        



if __name__ == '__main__':
    a = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    sys.exit(a.exec_())

