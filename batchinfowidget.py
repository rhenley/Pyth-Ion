# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'batchinfowidget.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_batchinfodialog(object):
    def setupUi(self, batchinfodialog):
        batchinfodialog.setObjectName(_fromUtf8("batchinfodialog"))
        batchinfodialog.resize(739, 819)
        self.gridLayout_2 = QtGui.QGridLayout(batchinfodialog)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.cancelbutton = QtGui.QPushButton(batchinfodialog)
        self.cancelbutton.setMaximumSize(QtCore.QSize(200, 75))
        self.cancelbutton.setObjectName(_fromUtf8("cancelbutton"))
        self.gridLayout_2.addWidget(self.cancelbutton, 1, 2, 1, 1)
        self.okbutton = QtGui.QPushButton(batchinfodialog)
        self.okbutton.setMaximumSize(QtCore.QSize(200, 75))
        self.okbutton.setObjectName(_fromUtf8("okbutton"))
        self.gridLayout_2.addWidget(self.okbutton, 1, 1, 1, 1)
        self.groupBox = QtGui.QGroupBox(batchinfodialog)
        self.groupBox.setAutoFillBackground(False)
        self.groupBox.setStyleSheet(_fromUtf8("QGroupBox { \n"
"     border: 2px solid gray; \n"
"     border-radius: 3px; \n"
" } "))
        self.groupBox.setFlat(False)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.gridLayout = QtGui.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label_6 = QtGui.QLabel(self.groupBox)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout.addWidget(self.label_6, 5, 0, 1, 1)
        self.label_2 = QtGui.QLabel(self.groupBox)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_3 = QtGui.QLabel(self.groupBox)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_4 = QtGui.QLabel(self.groupBox)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.label = QtGui.QLabel(self.groupBox)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.minfracbox = QtGui.QLineEdit(self.groupBox)
        self.minfracbox.setAlignment(QtCore.Qt.AlignCenter)
        self.minfracbox.setObjectName(_fromUtf8("minfracbox"))
        self.gridLayout.addWidget(self.minfracbox, 2, 1, 1, 1)
        self.sampratebox = QtGui.QLineEdit(self.groupBox)
        self.sampratebox.setAlignment(QtCore.Qt.AlignCenter)
        self.sampratebox.setObjectName(_fromUtf8("sampratebox"))
        self.gridLayout.addWidget(self.sampratebox, 0, 1, 1, 1)
        self.LPfilterbox = QtGui.QLineEdit(self.groupBox)
        self.LPfilterbox.setAlignment(QtCore.Qt.AlignCenter)
        self.LPfilterbox.setObjectName(_fromUtf8("LPfilterbox"))
        self.gridLayout.addWidget(self.LPfilterbox, 1, 1, 1, 1)
        self.mindwellbox = QtGui.QLineEdit(self.groupBox)
        self.mindwellbox.setAlignment(QtCore.Qt.AlignCenter)
        self.mindwellbox.setObjectName(_fromUtf8("mindwellbox"))
        self.gridLayout.addWidget(self.mindwellbox, 3, 1, 1, 1)
        self.cusumstepentry = QtGui.QLineEdit(self.groupBox)
        self.cusumstepentry.setAlignment(QtCore.Qt.AlignCenter)
        self.cusumstepentry.setObjectName(_fromUtf8("cusumstepentry"))
        self.gridLayout.addWidget(self.cusumstepentry, 4, 1, 1, 1)
        self.label_5 = QtGui.QLabel(self.groupBox)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)
        self.cusumthreshentry = QtGui.QLineEdit(self.groupBox)
        self.cusumthreshentry.setAlignment(QtCore.Qt.AlignCenter)
        self.cusumthreshentry.setObjectName(_fromUtf8("cusumthreshentry"))
        self.gridLayout.addWidget(self.cusumthreshentry, 5, 1, 1, 1)
        self.label_7 = QtGui.QLabel(self.groupBox)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.gridLayout.addWidget(self.label_7, 6, 0, 1, 1)
        self.minleveltbox = QtGui.QLineEdit(self.groupBox)
        self.minleveltbox.setAlignment(QtCore.Qt.AlignCenter)
        self.minleveltbox.setObjectName(_fromUtf8("minleveltbox"))
        self.gridLayout.addWidget(self.minleveltbox, 6, 1, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox, 0, 0, 1, 3)

        self.retranslateUi(batchinfodialog)
        QtCore.QMetaObject.connectSlotsByName(batchinfodialog)

    def retranslateUi(self, batchinfodialog):
        batchinfodialog.setWindowTitle(_translate("batchinfodialog", "Dialog", None))
        self.cancelbutton.setText(_translate("batchinfodialog", "Cancel", None))
        self.okbutton.setText(_translate("batchinfodialog", "OK", None))
        self.label_6.setText(_translate("batchinfodialog", "CUSUM Threshold:", None))
        self.label_2.setText(_translate("batchinfodialog", "Low-Pass Filter (kHz):", None))
        self.label_3.setText(_translate("batchinfodialog", "Min. Fractional Blockade:", None))
        self.label_4.setText(_translate("batchinfodialog", "Min. Dwell Time (μs):", None))
        self.label.setText(_translate("batchinfodialog", "Sampling Rate (kHz):", None))
        self.minfracbox.setText(_translate("batchinfodialog", "0.7", None))
        self.sampratebox.setText(_translate("batchinfodialog", "100", None))
        self.LPfilterbox.setText(_translate("batchinfodialog", "10", None))
        self.mindwellbox.setText(_translate("batchinfodialog", "10000", None))
        self.cusumstepentry.setText(_translate("batchinfodialog", ".25", None))
        self.label_5.setText(_translate("batchinfodialog", "CUSUM Step:", None))
        self.cusumthreshentry.setText(_translate("batchinfodialog", ".1", None))
        self.label_7.setText(_translate("batchinfodialog", "Min. Level time (μs)", None))
        self.minleveltbox.setText(_translate("batchinfodialog", "100", None))

