# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled_v1.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import os
from pathlib import Path
from image_processing import run_main
import pandas as pd
from setup import copy_settings_default, settings_file
from configparser import ConfigParser
import os


if not os.path.exists(settings_file):
    copy_settings_default()


config = ConfigParser()
config.read(settings_file)



class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(658, 600)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMinimumSize(QtCore.QSize(0, 0))
        self.centralwidget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.gridLayout.setContentsMargins(9, 9, 9, -1)
        self.gridLayout.setHorizontalSpacing(56)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 1, 0, 1, 1)
        self.Settings_widget = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Settings_widget.sizePolicy().hasHeightForWidth())
        self.Settings_widget.setSizePolicy(sizePolicy)
        self.Settings_widget.setMinimumSize(QtCore.QSize(0, 0))
        self.Settings_widget.setAutoFillBackground(False)
        self.Settings_widget.setStyleSheet("background-color: rgb(186, 189, 182);")
        self.Settings_widget.setObjectName("Settings_widget")
        self.Red_thresh_label = QtWidgets.QLabel(self.Settings_widget)
        self.Red_thresh_label.setGeometry(QtCore.QRect(10, 10, 171, 21))
        self.Red_thresh_label.setStyleSheet("background-color: rgb(239, 41, 41)")
        self.Red_thresh_label.setObjectName("Red_thresh_label")
        self.Red_thresh_box = QtWidgets.QSpinBox(self.Settings_widget)
        self.Red_thresh_box.setGeometry(QtCore.QRect(10, 40, 171, 26))
        self.Red_thresh_box.setStyleSheet("background-color: rgb(238, 238, 236);")
        self.Red_thresh_box.setMaximum(99999)
        self.Red_thresh_box.setObjectName("Red_thresh_box")
        self.Red_size_label = QtWidgets.QLabel(self.Settings_widget)
        self.Red_size_label.setGeometry(QtCore.QRect(10, 80, 171, 21))
        self.Red_size_label.setStyleSheet("background-color: rgb(239, 41, 41)")
        self.Red_size_label.setObjectName("Red_size_label")
        self.Red_size_box = QtWidgets.QSpinBox(self.Settings_widget)
        self.Red_size_box.setGeometry(QtCore.QRect(10, 110, 171, 26))
        self.Red_size_box.setStyleSheet("background-color: rgb(238, 238, 236);")
        self.Red_size_box.setMaximum(99999)
        self.Red_size_box.setObjectName("Red_size_box")
        self.Green_thresh_label = QtWidgets.QLabel(self.Settings_widget)
        self.Green_thresh_label.setGeometry(QtCore.QRect(10, 150, 171, 21))
        self.Green_thresh_label.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.Green_thresh_label.setObjectName("Green_thresh_label")
        self.Green_thresh_box = QtWidgets.QSpinBox(self.Settings_widget)
        self.Green_thresh_box.setGeometry(QtCore.QRect(10, 180, 171, 26))
        self.Green_thresh_box.setStyleSheet("background-color: rgb(238, 238, 236);")
        self.Green_thresh_box.setMaximum(99999)
        self.Green_thresh_box.setObjectName("Green_thresh_box")
        self.Green_size_label = QtWidgets.QLabel(self.Settings_widget)
        self.Green_size_label.setGeometry(QtCore.QRect(10, 220, 171, 21))
        self.Green_size_label.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.Green_size_label.setObjectName("Green_size_label")
        self.Green_size_box = QtWidgets.QSpinBox(self.Settings_widget)
        self.Green_size_box.setGeometry(QtCore.QRect(10, 250, 171, 26))
        self.Green_size_box.setStyleSheet("background-color: rgb(238, 238, 236);")
        self.Green_size_box.setMaximum(99999)
        self.Green_size_box.setObjectName("Green_size_box")
        self.Blue_thresh_label = QtWidgets.QLabel(self.Settings_widget)
        self.Blue_thresh_label.setGeometry(QtCore.QRect(10, 290, 171, 21))
        self.Blue_thresh_label.setStyleSheet("background-color:rgb(114, 159, 207)")
        self.Blue_thresh_label.setObjectName("Blue_thresh_label")
        self.Blue_thresh_box = QtWidgets.QSpinBox(self.Settings_widget)
        self.Blue_thresh_box.setGeometry(QtCore.QRect(10, 320, 171, 26))
        self.Blue_thresh_box.setStyleSheet("background-color: rgb(238, 238, 236);")
        self.Blue_thresh_box.setMaximum(99999)
        self.Blue_thresh_box.setObjectName("Blue_thresh_box")
        self.Blue_size_label = QtWidgets.QLabel(self.Settings_widget)
        self.Blue_size_label.setGeometry(QtCore.QRect(10, 360, 171, 21))
        self.Blue_size_label.setStyleSheet("background-color: rgb(114, 159, 207)")
        self.Blue_size_label.setObjectName("Blue_size_label")
        self.Blue_size_box = QtWidgets.QSpinBox(self.Settings_widget)
        self.Blue_size_box.setGeometry(QtCore.QRect(10, 390, 171, 26))
        self.Blue_size_box.setStyleSheet("background-color: rgb(238, 238, 236);")
        self.Blue_size_box.setMaximum(99999)
        self.Blue_size_box.setObjectName("Blue_size_box")

        self.Red_thresh_box.setValue(int(config["THRESHOLDS"]["Red"]))
        self.Red_size_box.setValue(int(config["SIZES"]["Red"]))
        self.Green_thresh_box.setValue(int(config["THRESHOLDS"]["Green"]))
        self.Green_size_box.setValue(int(config["SIZES"]["Green"]))
        self.Blue_thresh_box.setValue(int(config["THRESHOLDS"]["Blue"]))
        self.Blue_size_box.setValue(int(config["SIZES"]["Blue"]))
        self.gridLayout.addWidget(self.Settings_widget, 0, 0, 1, 3)

        self.container = QtWidgets.QFrame(self.centralwidget)
        self.container.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.container.sizePolicy().hasHeightForWidth())
        self.container.setSizePolicy(sizePolicy)
        self.container.setObjectName("container")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.container)
        self.gridLayout_2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout_2.setContentsMargins(0, 0, -1, 0)
        self.gridLayout_2.setHorizontalSpacing(0)
        self.gridLayout_2.setVerticalSpacing(10)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.Folder_label = QtWidgets.QLabel(self.container)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Folder_label.sizePolicy().hasHeightForWidth())
        self.Folder_label.setSizePolicy(sizePolicy)
        self.Folder_label.setMinimumSize(QtCore.QSize(0, 0))
        self.Folder_label.setAutoFillBackground(False)
        self.Folder_label.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-color: rgb(0, 0, 0);")
        self.Folder_label.setFrameShape(QtWidgets.QFrame.Box)
        self.Folder_label.setObjectName("Folder_label")
        self.gridLayout_2.addWidget(self.Folder_label, 0, 0, 1, 1)
        self.Files_List = QtWidgets.QListWidget(self.container)
        self.Files_List.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.Files_List.setMovement(QtWidgets.QListView.Static)
        self.Files_List.setResizeMode(QtWidgets.QListView.Fixed)
        # self.Files_List.setGridSize(QtCore.QSize(0, 0))
        self.Files_List.setViewMode(QtWidgets.QListView.ListMode)
        self.Files_List.setObjectName("Files_List")
        self.gridLayout_2.addWidget(self.Files_List, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.container, 0, 4, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 658, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionSet_Folder = QtWidgets.QAction(MainWindow)
        self.actionSet_Folder.setObjectName("actionSet_Folder")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.menuFile.addAction(self.actionSet_Folder)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.actionQuit.triggered.connect(MainWindow.close)
        self.actionSet_Folder.triggered.connect(self.user_get_dir)
        self.pushButton.clicked.connect(self.run_quant)

        self.statusbar.showMessage("Ready")
        self.progressbar = QtWidgets.QProgressBar(MainWindow)
        self.statusbar.addPermanentWidget(self.progressbar)
        # This is simply to show the bar
        self.progressbar.setGeometry(30, 40, 200, 25)
        self.progressbar.setValue(0)
        self.progressbar.setVisible(False)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Colocalizer"))
        self.pushButton.setText(_translate("MainWindow", "Run"))
        self.Red_thresh_label.setText(_translate("MainWindow", "Red Threshold"))
        self.Red_size_label.setText(_translate("MainWindow", "Red Minimum Size"))
        self.Green_thresh_label.setText(_translate("MainWindow", "Green Threshold"))
        self.Green_size_label.setText(_translate("MainWindow", "Green Minimum Size"))
        self.Blue_thresh_label.setText(_translate("MainWindow", "Blue Threshold"))
        self.Blue_size_label.setText(_translate("MainWindow", "Blue Minimum Size"))
        self.Folder_label.setText(_translate("MainWindow", "Folder"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionSet_Folder.setText(_translate("MainWindow", "Set Folder"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))



    def user_get_dir(self):
        # create folder selection dialog
        self.progressbar.setVisible(False)
        self.progressbar.setValue(0)
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.top_directory = QFileDialog.getExistingDirectory(self, caption="Select top level directory.",
                                                directory=config['DEFAULT']['last_dir'],
                                                options=options)
        config['DEFAULT']['last_dir'] = self.top_directory
        self.Folder_label.setText(self.top_directory)
        zvi_list = [str(file) for file in sorted(list(set([item for item in Path(self.top_directory).glob("*.czi")])))]
        # print(zvi_list)
        if zvi_list != []:
            self.Files_List.clear()
        self.Files_List.addItems(zvi_list)
        # print([self.Files_List.item(index).text() for index in range(self.Files_List.count())])
        with open(settings_file, 'w') as configfile:
            config.write(configfile)


    def run_quant(self):
        config["THRESHOLDS"]["Red"] = str(self.Red_thresh_box.value())
        config["SIZES"]['Red'] = str(self.Red_size_box.value())
        config["THRESHOLDS"]["Green"] = str(self.Green_thresh_box.value())
        config["SIZES"]['Green'] = str(self.Green_size_box.value())
        config["THRESHOLDS"]["Blue"] = str(self.Blue_thresh_box.value())
        config["SIZES"]['Blue'] = str(self.Blue_size_box.value())
        with open(settings_file, 'w') as configfile:
            config.write(configfile)
        self.statusbar.showMessage("Running...")
        assert self.Files_List != []
        files = [self.Files_List.item(index).text() for index in range(self.Files_List.count())]
        self.progressbar.setMaximum(len(files))
        self.progressbar.setVisible(True)
        for iii, item in enumerate(files):
            this = self.Files_List.item(iii)
            this.setSelected(True)
            self.progressbar.setValue(iii + 1)
            run_main(item, self.Red_thresh_box.value(), self.Red_size_box.value(),
                     self.Green_thresh_box.value(), self.Green_size_box.value(),
                     self.Blue_thresh_box.value(), self.Blue_size_box.value())
        self.statusbar.showMessage("Done!")
