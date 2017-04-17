import sys

from PyQt5.QtWidgets import QApplication

from wat import WaveletAnalysisApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = WaveletAnalysisApp()
    form.show()
    sys.exit(app.exec_())
