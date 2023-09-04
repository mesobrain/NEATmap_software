from Environment_ui import *
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QDialog, QHeaderView, QTableWidgetItem, QTableWidget
from PySide2.QtUiTools import QUiLoader
from pages.Settings.default_param import default_param

class Settings(QDialog):
    def __init__(self):
        super(Settings, self).__init__()
        self.settings = QUiLoader().load('pages/Settings/setting.ui')
        self.settings.Params.setRowCount(33)
        self.settings.Params.setColumnCount(2)
        self.settings.Params.setHorizontalHeaderLabels(['Params name', 'Value'])
        self.settings.Params.horizontalHeader().setStyleSheet("QHeaderView {font-size: 18pt};")
        self.settings.Params.horizontalHeader().setStyleSheet("::section {background-color: lightGray;font-size:16pt;}")
        self.settings.Params.horizontalHeader().setStretchLastSection(True)
        self.settings.Params.verticalHeader().setStretchLastSection(True)
        self.settings.Params.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.settings.Params.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.params = default_param
        self.params_name = list(self.params.keys())
        values = list(self.params.values())
        for i in range(len(self.params_name)):
            params_group = []
            params_group.append(self.params_name[i]) 
            params_group.append(values[i])
            for j in range(2):
                self.settings.Params.setItem(i, j, QTableWidgetItem(params_group[j]))

        for i in range(self.settings.Params.rowCount()):
            for j in range(self.settings.Params.columnCount()):
                if self.settings.Params.item(i, j):
                    self.settings.Params.item(i, j).setTextAlignment(Qt.AlignCenter)

        self.settings.Params.cellChanged.connect(self.cell_change)

        self.settings.Save.clicked.connect(self.update_params)

    def cell_change(self, row, column):
        self.settings.Params.item(row, column).setTextAlignment(Qt.AlignCenter)

    def update_params(self):
        updata_value = []
        for i in range(len(self.params_name)):
            v = self.settings.Params.item(i, 1).text()
            updata_value.append(v)
        for k in range(len(self.params_name)):
            self.params[self.params_name[k]] = updata_value[k]

        return self.params
