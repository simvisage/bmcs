import sys

from PySide import QtCore, QtGui


def main():

    app = QtGui.QApplication(sys.argv)

    curTime = QtCore.QTime.currentTime()
    print(curTime)

    label = QtGui.QLabel()
    pixmap = QtGui.QPixmap('scales.jpg')
    label.setPixmap(pixmap)
    label.move(700, 400)  # Screen X,Y position
    label.setWindowFlags(QtCore.Qt.SplashScreen |
                         QtCore.Qt.WindowStaysOnTopHint)
    label.show()

    QtCore.QTimer.singleShot(5000, app.quit)

    app.exec_()


if __name__ == '__main__':
    main()
