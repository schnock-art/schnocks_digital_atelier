import sys
import time
from PyQt6.QtWidgets import QApplication, QWidget, QProgressBar, QPushButton, \
    QHBoxLayout, QVBoxLayout
from PyQt6.QtGui import QIcon


class ProgressBar(QProgressBar):
    def __init__(self, parent=None, max_value=100):
        super().__init__(parent)
        self.max_value = max_value
        self.setMaximum(max_value)
        self._active = True

    def updateBar(self, value):
        self.setValue(value)

        if value >= int(self.max_value/2):
            self.changeColor('green')

        if value >= self.maximum():
            self.setValue(self.max_value)
            self.active = False

    def changeColor(self, color):
        css = """
            ::chunk {{
                background: {0};
            }}
        """.format(color)
        self.setStyleSheet(css)
