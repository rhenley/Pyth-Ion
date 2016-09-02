# -*- coding: utf8 -*-
import sys
import numpy as np
from batchinfowidget import *

class batchprocesser(QtGui.QWidget):


    def __init__(self, master=None):
        QtGui.QWidget.__init__(self,master)
        self.uibp = Ui_batchinfodialog()
        self.uibp.setupUi(self)
        
        QtCore.QObject.connect(self.uibp.cancelbutton, QtCore.SIGNAL('clicked()'), self.close)
        
    def close(self):
        self.destroy()
    
    
if __name__ == "__main__":
    global myapp_bp
    app_bp = QtGui.QApplication(sys.argv)
    myapp_bp = batchprocesser()
    myapp_bp.show()
    sys.exit(app_bp.exec_())