from PyQt5.QtWidgets import QApplication, QDialog
import sys

from .appframe import AppFrame

# 创建一个应用
app = QApplication(sys.argv)  # app = QApplication([])
# 创建一个窗体
dlg = AppFrame()
# 进入消息循环
status = app.exec()
# 返回状态码给系统
sys.exit(status)
