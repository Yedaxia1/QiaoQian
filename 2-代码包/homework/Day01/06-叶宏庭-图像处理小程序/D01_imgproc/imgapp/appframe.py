from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QImage, QPixmap
from .procui import Ui_proc
import cv2

class AppFrame(QDialog):
    def __init__(self):
        super(AppFrame,self).__init__()
        self.ui = Ui_proc()
        self.ui.setupUi(self)
        # 加载图像
        self.img = cv2.imread("./imgapp/gpu.bmp")
        self.show_img(self.img)
        self.show()
        
    

    def show_img(self, src_img):
        # 转换为Qt中的图像
        q_img = QImage(src_img.tobytes(), 
        src_img.shape[1], src_img.shape[0], 
        src_img.shape[1]*src_img.shape[2], 
        QImage.Format_BGR888
        )
        # 转换为像素图
        q_pixmap = QPixmap.fromImage(q_img)
        # 显示
        self.ui.lbl_img.setPixmap(q_pixmap)
        self.ui.lbl_img.setScaledContents(True)


    def img_ori(self):
        self.show_img(self.img)

    def img_sobel_x(self):
        # 图像特征处理
        i_sobel = cv2.Sobel(self.img, -1, 1, 0, scale=1, delta=0)
        # 显示
        self.show_img(i_sobel)

    def img_soble_y(self):
        # 图像特征处理
        i_sobel = cv2.Sobel(self.img, -1, 0, 1, scale=1, delta=0)
        # 显示
        self.show_img(i_sobel)
    
    def img_sobel_xy(self):
        # 图像特征处理
        i_sobel = cv2.Sobel(self.img, -1, 1, 1, scale=1, delta=0)
        # 显示
        self.show_img(i_sobel)

    def img_Laplace(self):
        # 图像特征处理
        i_sobel = cv2.Laplacian(self.img, -1, scale=1, delta=0)
        # 显示
        self.show_img(i_sobel)
    