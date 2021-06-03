# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'proc.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_proc(object):
    def setupUi(self, proc):
        proc.setObjectName("proc")
        proc.resize(657, 363)
        proc.setStyleSheet("QPushButton{\n"
"    \n"
"    border-width:2px;\n"
"    border-style:solid;\n"
"    border-radius:8px;\n"
"    border-top-color:#FFFFFF;\n"
"    border-right-color:#FFFFFF;\n"
"    border-bottom-color:#888888;\n"
"    border-left-color:#888888;\n"
"}")
        self.lbl_img = QtWidgets.QLabel(proc)
        self.lbl_img.setGeometry(QtCore.QRect(10, 10, 481, 351))
        self.lbl_img.setFrameShape(QtWidgets.QFrame.Panel)
        self.lbl_img.setFrameShadow(QtWidgets.QFrame.Raised)
        self.lbl_img.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_img.setObjectName("lbl_img")
        self.btn_ori = QtWidgets.QPushButton(proc)
        self.btn_ori.setGeometry(QtCore.QRect(530, 70, 93, 28))
        self.btn_ori.setObjectName("btn_ori")
        self.btn_soble_x = QtWidgets.QPushButton(proc)
        self.btn_soble_x.setGeometry(QtCore.QRect(530, 170, 93, 28))
        self.btn_soble_x.setObjectName("btn_soble_x")
        self.sobel_y = QtWidgets.QPushButton(proc)
        self.sobel_y.setGeometry(QtCore.QRect(530, 220, 93, 28))
        self.sobel_y.setObjectName("sobel_y")
        self.sobel_x_y = QtWidgets.QPushButton(proc)
        self.sobel_x_y.setGeometry(QtCore.QRect(530, 270, 91, 31))
        self.sobel_x_y.setObjectName("sobel_x_y")
        self.Laplace.setGeometry(QtCore.QRect(530, 280, 91, 31))
        self.Laplace.setObjectName("Laplace")
        
        self.retranslateUi(proc)
        self.btn_ori.clicked.connect(proc.img_ori)
        self.btn_soble_x.clicked.connect(proc.img_sobel_x)
        self.sobel_y.clicked.connect(proc.img_soble_y)
        self.sobel_x_y.clicked.connect(proc.img_sobel_xy)
        self.Laplace.clicked.connect(proc.img_Laplace)
        QtCore.QMetaObject.connectSlotsByName(proc)

    def retranslateUi(self, proc):
        _translate = QtCore.QCoreApplication.translate
        proc.setWindowTitle(_translate("proc", "图像处理"))
        self.lbl_img.setText(_translate("proc", "<font color=red>图像显示</font>"))
        self.btn_ori.setText(_translate("proc", "原图"))
        self.btn_soble_x.setText(_translate("proc", "Sobel-x-1"))
        self.sobel_y.setText(_translate("proc", "Sobel-y-1"))
        self.sobel_x_y.setText(_translate("proc", "Sobel-x-y-1"))
        self.Laplace.setText(_translate("proc", "Laplace"))
