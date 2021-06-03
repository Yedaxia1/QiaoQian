from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QImage, QPixmap
from .appUI import Ui_Image_Dialog
import cv2

class AppFrame(QDialog):
    def __init__(self):
        super(AppFrame, self).__init__()
        self.UI = Ui_Image_Dialog()
        self.UI.setupUi(self)

        self.img = cv2.imread('./imgApp/Test.jpg')
        self.show_img(self.img)
        self.show()
    
    def show_img(self, img):
        # 转换为Qt的图像
        q_img = QImage(
            img.tobytes(), 
            img.shape[1], img.shape[0], 
            img.shape[1] * img.shape[2], 
            QImage.Format_BGR888
        ) 
        # 转换为像素图
        q_pixmap = QPixmap.fromImage(q_img)
        # 显示
        self.UI.ImageLabel.setPixmap(q_pixmap)
        self.UI.ImageLabel.setScaledContents(True)
    
    def img_ori(self):
        self.show_img(self.img)
    
    def img_gaussian(self):
        image = cv2.GaussianBlur(self.img, (15, 15), 10)
        self.show_img(image)
    
    def img_remove_R(self):
        image = self.img.copy()
        image[::, ::, 2] = 0
        self.show_img(image)
    
    def img_remove_G(self):
        image = self.img.copy()
        image[::, ::, 1] = 0
        self.show_img(image)
    
    def img_remove_B(self):
        image = self.img.copy()
        image[::, ::, 0] = 0
        self.show_img(image)
    
    def img_laplace(self):
        image = cv2.Laplacian(self.img, cv2.CV_16S, ksize=3)
        image = cv2.convertScaleAbs(image)
        self.show_img(image)

    def img_sobel_x(self):
        image = cv2.Sobel(self.img, -1, 1, 0)
        self.show_img(image)
    
    def img_sobel_y(self):
        image = cv2.Sobel(self.img, -1, 0, 1)
        self.show_img(image)
    
    def img_sobel_xy(self):
        image = cv2.Sobel(self.img, -1, 1, 1)
        self.show_img(image)
