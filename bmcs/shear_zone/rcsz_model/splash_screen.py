import sys
from _thread import start_new_thread
import time

from PySide import QtCore, QtGui


def splash_screen(image_file='scales.jpg', seconds=5):

    app = QtGui.QApplication.instance()
    curTime = QtCore.QTime.currentTime()
    print(curTime)

    pixmap = QtGui.QPixmap("scales.jpg")
    pixmap = pixmap.scaled(5000, 600, QtCore.Qt.KeepAspectRatio)

    myLabel = QtGui.QLabel()
    myLabel.setPixmap(pixmap)
#    myLabel.move(200, 100)  # Screen X,Y position

    myLabel.setWindowFlags(QtCore.Qt.SplashScreen |
                           QtCore.Qt.WindowStaysOnTopHint)
    myLabel.setScaledContents(True)
    myLabel.show()

    QtCore.QTimer.singleShot(seconds * 1000, app.quit)

    app.exec_()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    rr = start_new_thread(splash_screen, ())
    time.sleep(10)
