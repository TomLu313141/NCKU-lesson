from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()
from PIL import Image
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QDialog,QGridLayout, QLabel, QPushButton
from PyQt5 import QtCore, QtGui
from matplotlib.image import imread
import math
import tensorflow as tf
from tensorflow.keras import datasets, models
import random
from torchsummary import summary
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.models as models
from torchvision import models


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


        self.loadImage = QPushButton('5.1 Show Train Image', self)
        self.loadImage.setGeometry(QtCore.QRect(20, 130, 201, 32))
        self.loadImage.setFont(font)
        
        self.loadImage.clicked.connect(self.show_image)
        ##########
    

        self.colorSep = QPushButton('5.2 Show HyperParameter', self)
        self.colorSep.setGeometry(QtCore.QRect(20, 210, 201, 32))
        self.colorSep.setFont(font)

        self.colorSep.clicked.connect(self.show_hyper)
        ##########
        

        self.colorTran = QPushButton('5.3 Show Model Shortcut', self)
        self.colorTran.setGeometry(QtCore.QRect(20, 290, 201, 32))
        self.colorTran.setFont(font)

        self.colorTran.clicked.connect(self.show_model)
        ##########
 

        self.Blending = QPushButton('5.4 Show Accuracy', self)
        self.Blending.setGeometry(QtCore.QRect(20, 370, 201, 32))
        self.Blending.setFont(font)

        self.Blending.clicked.connect(self.show_acc)
        ##########


    def show_image(self):
        (X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()

        y_train = y_train.reshape(-1,)
        y_train[:5]
        y_test = y_test.reshape(-1,)
        classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        a = [0]+[random.randint(0,9000) for _ in range(9)]
        for i in range(1,10):
            plt.subplot(3,3,i)
            
            index = a[i] 

            plt.imshow(X_train[index].reshape(32, 32, 3))
            plt.title(classes[y_train[index]],size=10)
            plt.xticks([])
            plt.yticks([])

        plt.show()

    def show_hyper(self):
        '''定义超参数'''
        batch_size = 32       
        learning_rate = 1e-3    
        # num_epoches = 30        

        print('hyperparameters: ')
        print(f"batch size: {batch_size:02d}")
        print(f"learning rate: {learning_rate:.3f}")
        print("ottimizer: SGD")


    def show_model(self):
        vgg16 = models.vgg16(pretrained=True)
        vgg16.classifier[6] = nn.Linear(4096,2)
        vgg16 = vgg16.cuda() #use GPU

        summary(vgg16.cuda(), (3, 224, 224))

    def show_acc(self):
        img_address = 'C:/Users/s1551/Desktop/Hw1/acc.jpg'
    
        img = cv2.imread(img_address)
        cv2.imshow('Accuracy', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    a = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    sys.exit(a.exec_())

