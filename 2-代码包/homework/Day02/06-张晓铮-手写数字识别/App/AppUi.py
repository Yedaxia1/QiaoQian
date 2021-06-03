# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\App.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Predicate(object):
    def setupUi(self, Predicate):
        Predicate.setObjectName("Predicate")
        Predicate.resize(508, 424)
        self.layoutWidget = QtWidgets.QWidget(Predicate)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 30, 481, 381))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.CamLabel = QtWidgets.QLabel(self.layoutWidget)
        self.CamLabel.setText("")
        self.CamLabel.setObjectName("CamLabel")
        self.verticalLayout.addWidget(self.CamLabel)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.open_button = QtWidgets.QPushButton(self.layoutWidget)
        self.open_button.setObjectName("open_button")
        self.horizontalLayout.addWidget(self.open_button)
        self.cap_button = QtWidgets.QPushButton(self.layoutWidget)
        self.cap_button.setObjectName("cap_button")
        self.horizontalLayout.addWidget(self.cap_button)
        self.pre_button = QtWidgets.QPushButton(self.layoutWidget)
        self.pre_button.setObjectName("pre_button")
        self.horizontalLayout.addWidget(self.pre_button)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Anonymice Powerline")
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.res_label = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Anonymice Powerline")
        font.setPointSize(16)
        self.res_label.setFont(font)
        self.res_label.setText("")
        self.res_label.setObjectName("res_label")
        self.horizontalLayout_2.addWidget(self.res_label)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Anonymice Powerline")
        font.setPointSize(16)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.prob_label = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Anonymice Powerline")
        font.setPointSize(16)
        self.prob_label.setFont(font)
        self.prob_label.setText("")
        self.prob_label.setObjectName("prob_label")
        self.horizontalLayout_2.addWidget(self.prob_label)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout.setStretch(0, 5)

        self.retranslateUi(Predicate)
        self.open_button.clicked.connect(Predicate.begin)
        self.cap_button.clicked.connect(Predicate.cap)
        self.pre_button.clicked.connect(Predicate.predicate)
        QtCore.QMetaObject.connectSlotsByName(Predicate)

    def retranslateUi(self, Predicate):
        _translate = QtCore.QCoreApplication.translate
        Predicate.setWindowTitle(_translate("Predicate", "Dialog"))
        self.open_button.setText(_translate("Predicate", "开始摄像"))
        self.cap_button.setText(_translate("Predicate", "拍摄"))
        self.pre_button.setText(_translate("Predicate", "预测"))
        self.label_2.setText(_translate("Predicate", "结果："))
        self.label_4.setText(_translate("Predicate", "置信度："))
