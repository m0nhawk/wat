import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from wat import WaveletAnalysisApp

# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook


def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


# Set the exception hook to our wrapping function
sys.excepthook = my_exception_hook

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = WaveletAnalysisApp()
    form.show()
    try:
        sys.exit(app.exec_())
    except:
        print("Exiting")
