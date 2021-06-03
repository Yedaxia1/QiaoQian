# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\app.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Image_Dialog(object):
    def setupUi(self, Image_Dialog):
        Image_Dialog.setObjectName("Image_Dialog")
        Image_Dialog.resize(583, 408)
        self.widget = QtWidgets.QWidget(Image_Dialog)
        self.widget.setGeometry(QtCore.QRect(21, 20, 531, 371))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.ImageLabel = QtWidgets.QLabel(self.widget)
        self.ImageLabel.setText("")
        self.ImageLabel.setObjectName("ImageLabel")
        self.horizontalLayout.addWidget(self.ImageLabel)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.push_ori = QtWidgets.QPushButton(self.widget)
        self.push_ori.setObjectName("push_ori")
        self.verticalLayout.addWidget(self.push_ori)
        self.push_Gaussian = QtWidgets.QPushButton(self.widget)
        self.push_Gaussian.setObjectName("push_Gaussian")
        self.verticalLayout.addWidget(self.push_Gaussian)
        self.push_R = QtWidgets.QPushButton(self.widget)
        self.push_R.setObjectName("push_R")
        self.verticalLayout.addWidget(self.push_R)
        self.push_G = QtWidgets.QPushButton(self.widget)
        self.push_G.setObjectName("push_G")
        self.verticalLayout.addWidget(self.push_G)
        self.push_B = QtWidgets.QPushButton(self.widget)
        self.push_B.setObjectName("push_B")
        self.verticalLayout.addWidget(self.push_B)
        self.push_Laplace = QtWidgets.QPushButton(self.widget)
        self.push_Laplace.setObjectName("push_Laplace")
        self.verticalLayout.addWidget(self.push_Laplace)
        self.push_Sobel_x = QtWidgets.QPushButton(self.widget)
        self.push_Sobel_x.setObjectName("push_Sobel_x")
        self.verticalLayout.addWidget(self.push_Sobel_x)
        self.push_Sobel_y = QtWidgets.QPushButton(self.widget)
        self.push_Sobel_y.setObjectName("push_Sobel_y")
        self.verticalLayout.addWidget(self.push_Sobel_y)
        self.push_Sobel_xy = QtWidgets.QPushButton(self.widget)
        self.push_Sobel_xy.setObjectName("push_Sobel_xy")
        self.verticalLayout.addWidget(self.push_Sobel_xy)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.horizontalLayout.setStretch(0, 3)
        self.horizontalLayout.setStretch(1, 1)

        self.retranslateUi(Image_Dialog)
        self.push_ori.clicked.connect(Image_Dialog.img_ori)
        self.push_Gaussian.clicked.connect(Image_Dialog.img_gaussian)
        self.push_R.clicked.connect(Image_Dialog.img_remove_R)
        self.push_G.clicked.connect(Image_Dialog.img_remove_G)
        self.push_B.clicked.connect(Image_Dialog.img_remove_B)
        self.push_Laplace.clicked.connect(Image_Dialog.img_laplace)
        self.push_Sobel_x.clicked.connect(Image_Dialog.img_sobel_x)
        self.push_Sobel_y.clicked.connect(Image_Dialog.img_sobel_y)
        self.push_Sobel_xy.clicked.connect(Image_Dialog.img_sobel_xy)
        QtCore.QMetaObject.connectSlotsByName(Image_Dialog)

    def retranslateUi(self, Image_Dialog):
        _translate = QtCore.QCoreApplication.translate
        Image_Dialog.setWindowTitle(_translate("Image_Dialog", "Dialog"))
        self.push_ori.setText(_translate("Image_Dialog", "原图"))
        self.push_Gaussian.setText(_translate("Image_Dialog", "高斯模糊"))
        self.push_R.setText(_translate("Image_Dialog", "R通道置零"))
        self.push_G.setText(_translate("Image_Dialog", "G通道置零"))
        self.push_B.setText(_translate("Image_Dialog", "B通道置零"))
        self.push_Laplace.setText(_translate("Image_Dialog", "Laplace"))
        self.push_Sobel_x.setText(_translate("Image_Dialog", "Sobel-x"))
        self.push_Sobel_y.setText(_translate("Image_Dialog", "Sobel-y"))
        self.push_Sobel_xy.setText(_translate("Image_Dialog", "Sobel-xy"))
