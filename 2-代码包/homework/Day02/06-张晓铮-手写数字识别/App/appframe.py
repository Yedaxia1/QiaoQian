from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from .AppUi import Ui_Predicate
import cv2
from PyQt5 import QtWidgets
import torch
import torch.nn as nn
from Model.SimpleNet import simplenet

class AppFrame(QDialog):
    def __init__(self):
        super(AppFrame, self).__init__()
        self.Ui = Ui_Predicate()
        self.Ui.setupUi(self)
        self.timer = QTimer()
        self.video = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.timer.timeout.connect(self.show_cam)
        self.net = simplenet()
        self.net.load_state_dict(torch.load('./App/model.pth'))
        self.net.eval()
        self.net = self.net.to('cuda')
        self.curr_img = None
        self.show()
    
    def begin(self):
        if not self.timer.isActive():
            flag = self.video.open(self.CAM_NUM)
            self.curr_img = None
            self.Ui.res_label.setText('')
            self.Ui.prob_label.setText('')
            if flag:
                self.timer.start(30)
            else:
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            
    def show_cam(self):
        flag, img = self.video.read()

        q_img = QImage(
            img.tobytes(), 
            img.shape[1], img.shape[0], 
            img.shape[1] * img.shape[2], 
            QImage.Format_BGR888
        ) 
        q_pixmap = QPixmap.fromImage(q_img)
        self.Ui.CamLabel.setPixmap(q_pixmap)
        self.Ui.CamLabel.setScaledContents(True)
    
    def cap(self):
        if self.timer.isActive():
            flag, img = self.video.read()
            self.timer.stop()

            q_img = QImage(
                img.tobytes(), 
                img.shape[1], img.shape[0], 
                img.shape[1] * img.shape[2], 
                QImage.Format_BGR888
            ) 
            q_pixmap = QPixmap.fromImage(q_img)
            self.Ui.CamLabel.setPixmap(q_pixmap)
            self.Ui.CamLabel.setScaledContents(True)

            self.curr_img = img

    def predicate(self):
        if not self.timer.isActive():
            img = self.curr_img
            img = cv2.resize(img, (32, 32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype('float32')
            img = img / 255.0

            img_tensor = torch.from_numpy(img).clone()
            
            res = self.net(img_tensor.view(-1, 1, 32, 32).to('cuda'))
            index = res.argmax(dim=1).cpu().item()
            softmax = nn.Softmax()
            prob = softmax(res)
            self.Ui.prob_label.setText(str(prob.cpu()[0, index].item()))
            self.Ui.res_label.setText(str(index))


            
            


