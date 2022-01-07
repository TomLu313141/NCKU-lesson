from __future__ import print_function
from __future__ import division

import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.image as mpimg
import seaborn as sns; sns.set()
from keras.models import load_model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import tensorflow as tf
import random

from PyQt5.QtWidgets import QApplication, QDialog, QGridLayout, QLabel, QPushButton
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QPushButton, QLabel, QInputDialog)

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


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

        #5-1
        self.loadImage = QPushButton('5.1 Show Model', self)
        self.loadImage.setGeometry(QtCore.QRect(20, 130, 201, 32))
        self.loadImage.setFont(font)
        self.loadImage.clicked.connect(self.show_model)
        # ##########
    
        #5-2
        self.colorSep = QPushButton('5.2 Show Tensorboard', self)
        self.colorSep.setGeometry(QtCore.QRect(20, 210, 201, 32))
        self.colorSep.setFont(font)
        self.colorSep.clicked.connect(self.show_ten)
        ##########
        

        # 5-4
        self.Blending = QPushButton('5.4 argumentation', self)
        self.Blending.setGeometry(QtCore.QRect(20, 370, 201, 32))
        self.Blending.setFont(font)
        self.Blending.clicked.connect(self.argument)
        ##########

        # 5-3
        self.Test = QPushButton('5.3 Test', self)
        self.Test.setGeometry(QtCore.QRect(20, 290, 201, 32))
        self.Test.setFont(font)
        self.Test.clicked.connect(self.showDialog)

    # 5-3
    def showDialog(self):
        sender = self.sender()
      
        if sender == self.Test:
            text, ok = QInputDialog.getText(self, '編號', 'enter a number：')
            if ok:

                index = int(text)

                model = load_model("C:/Users/s1551/Desktop/model-resnet50-final_2.h5")
                cls_list = ['cats', 'dogs']
                label_dict = {0: 'cat', 1: 'dog'}  # 設定標籤

                i=0
                for filename in os.listdir(r"C:/Users/s1551/Desktop/train"):
                    i=i+1
                    if i==index:
                        img = cv2.imread("C:/Users/s1551/Desktop/train/"+filename)
                        plt.imshow(img)
                        break
                img=cv2.resize(img,(224,224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                prob = model.predict(x)
                for i in range(0,prob.shape[0]):
                    prediction = np.argmax(prob[i])
                plt.title("Class:"+ label_dict[prediction])
                plt.grid(False)
                plt.show()
    # 5-1
    def show_model(self):
        model = load_model("C:/Users/s1551/Desktop/model-resnet50-final_2.h5")
        model.summary()

    # 5-3
    def show_ten(self):
        img= cv2.imread('C:/Users/s1551/Desktop/HW2/result.jpg')
        cv2.imshow('Tensorboard', img)

    # 5-4
    def argument(self):
        data = ['Before Random-Erasing','After Random-Erasing']
        acc = [65.3,70.1]

        x = np.arange(len(data))
        plt.bar(x,acc)
        plt.xticks(x,data)

        plt.ylabel("Accuracy")
        plt.grid(axis = "y")
        plt.ylim(60,80)
        plt.savefig("comparision")
        plt.show()


if __name__ == '__main__':
    a = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    sys.exit(a.exec_())