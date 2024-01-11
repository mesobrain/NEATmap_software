from Environment_ui import *
from PySide2.QtWidgets import QWidget, QFileDialog, QTextBrowser, QPushButton, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import Signal, QObject
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from pages.Data_preprocessing.cutting import single_cutting

class SignalStore(QObject):
    progress_update = Signal(int)
    progress_cut_update = Signal(int)
    text_print = Signal(QTextBrowser, str)
    splice_print = Signal(QTextBrowser, str)
    stop_text = Signal(QPushButton)
    stop_cut = Signal(QPushButton)

class Datapreprocess(QWidget):
    def __init__(self, params):
        super(Datapreprocess, self).__init__()
        self.so = SignalStore()
        self.so.progress_update.connect(self.setProgress)
        self.so.progress_cut_update.connect(self.setProgressCut)
        self.so.text_print.connect(self.printToGui)
        self.so.splice_print.connect(self.printToGui)
        self.so.stop_text.connect(self.check_stop)
        self.so.stop_cut.connect(self.check_stop)
        self.data_preprocess = QUiLoader().load('pages/Data_preprocessing/data_preprocessing.ui')

        self.data_preprocess.dataline.setPlaceholderText("Select BrainImgae/4.0 for the reconstruction data path")
        self.data_preprocess.saveline.setPlaceholderText("Select the path to save the data")
        self.data_preprocess.stop.setEnabled(False)
        self.data_preprocess.stopCut.setEnabled(False)

        self.data_preprocess.load.clicked.connect(self.SeletData)
        self.data_preprocess.selet.clicked.connect(self.SeletSave)
        self.data_preprocess.stop.clicked.connect(self.change)
        self.data_preprocess.stopCut.clicked.connect(self.stop_cut)

        self.data_preprocess.comboBox561.addItems(['C1', 'C2', 'C3', 'C4'])
        self.data_preprocess.comboBox488.addItems(['C1', 'C2', 'C3', 'C4'])
        self.data_preprocess.comboBox405.addItems(['C1', 'C2', 'C3', 'C4'])

        self.data_preprocess.start.clicked.connect(self.run_datapreprocess)
        self.data_preprocess.startCut.clicked.connect(self.Run)
        self.data_preprocess.Brain2dto3dTextEdit.document().setMaximumBlockCount(10000)
        self.data_preprocess.progressBar.setMaximum(100)
        self.data_preprocess.progressBarCutting.setMaximum(100)
        
        self.params = params

    def SeletData(self):
        filepath = QFileDialog.getExistingDirectory(self.data_preprocess, 'Selet root')
        self.data_preprocess.dataline.setText(filepath)
    
    def SeletSave(self):
        filepath = QFileDialog.getExistingDirectory(self.data_preprocess, 'Selet root')
        self.data_preprocess.saveline.setText(filepath)

    def run_datapreprocess(self):

        if len(self.data_preprocess.dataline.text()) == 0:
            QMessageBox.critical(
                self.data_preprocess,
                "Error",
                "Please select data path."
            )
        elif len(self.data_preprocess.saveline.text()) == 0:
            QMessageBox.critical(
                self.data_preprocess,
                "Error",
                "Please select data save path."
            )
        elif self.data_preprocess.radioButton561.isChecked() == False and self.data_preprocess.radioButton488.isChecked() == False and self.data_preprocess.radioButton405.isChecked() == False:
            QMessageBox.critical(
                self.data_preprocess,
                "Error",
                "Please select channel."
            )

        json_path = os.path.join(self.data_preprocess.dataline.text(), '..', 'freesia_4.0_'+ self.data_preprocess.comboBox405.currentText() + '_405nm_10X.json')
        with open(json_path) as f:
            brain = json.load(f)
            images = brain['images']
            total_num = len(images)
            
        if self.data_preprocess.buttonGroup.checkedButton().text() == '561nm':
            select_channel = self.data_preprocess.comboBox561.currentText()
        elif self.data_preprocess.buttonGroup.checkedButton().text() == '488nm':
            select_channel = self.data_preprocess.comboBox488.currentText()
        elif self.data_preprocess.buttonGroup.checkedButton().text() == '405nm':
            select_channel = self.data_preprocess.comboBox405.currentText()

        thread = Thread(target=self.brain2dto3d, args=(total_num, select_channel))
        self.data_preprocess.start.setEnabled(False)
        self.data_preprocess.stop.setEnabled(True)
        thread.start()
        

    def printToGui(self, fb, text):
        fb.appendPlainText(text)

    def setProgress(self, value):
        self.data_preprocess.progressBar.setValue(value)

    def change(self):
        self.data_preprocess.stop.setText('Stopping')

    def stop_cut(self):
        self.data_preprocess.stopCut.setText('Stopping')

    def check_stop(self, sp):
        if sp.text() == 'Stopping':
            os._exit(0)

    def setProgressCut(self, value):
        self.data_preprocess.progressBarCutting.setValue(value)
        
    def brain2dto3d(self, total_num, select_channel):
        z_num = int(self.params['Network_input_z'])
        name_index = 1
        temp = []
        j = 0
        save_path = self.data_preprocess.saveline.text() + '/brain_image_64_' + self.data_preprocess.buttonGroup.checkedButton().text()
        os.makedirs(save_path, exist_ok=True)
        for i in range(0, total_num):
            self.so.stop_text.emit(self.data_preprocess.stop)
            self.so.progress_update.emit(int(((i+1) / total_num) * 100))
            image = sitk.ReadImage(os.path.join(self.data_preprocess.dataline.text(), 'Z{:05d}_'.format(i) + select_channel +'.tif'))
            array = sitk.GetArrayFromImage(image)
            temp.append(array)
            if i == z_num - 1 + j:
                tif = np.array(temp)
                tif = sitk.GetImageFromArray(tif)
                sitk.WriteImage(tif, os.path.join(save_path, 'Z{:05d}.tif'.format(name_index)))
                j += z_num
                self.so.text_print.emit(self.data_preprocess.Brain2dto3dTextEdit, 'Finished {} image'.format(name_index))
                name_index += 1
                temp = []
        self.data_preprocess.start.setEnabled(True)
        self.data_preprocess.stop.setText('Finished')
    
    def Cutting(self, root, cut_size, channel, cut_index_x, cut_index_y, patch_weight_num, 
            patch_hegiht_num, index, train_path=None, label_path=None, cut_label=False):
        if cut_label:
            data_path = os.path.join(root, 'brain_label_64_' + channel)
            save_path = os.path.join(root, 'PatchSeg_' + channel)
        else:
            data_path = os.path.join(root, 'brain_image_64_' + channel)
            save_path = os.path.join(root, 'PatchImage_' + channel)
        if train_path is not None:
            save_path = os.path.join(root, 'train_image')
        if label_path is not None:
            save_path = os.path.join(root, 'train_label')

        self.so.stop_cut.emit(self.data_preprocess.stopCut)
        if cut_label:
            name = 'Z{:05d}_seg'.format(index)
        else:
            name = 'Z{:05d}'.format(index)
        image = os.path.join(data_path, name + '.tif')
        single_cutting(image, index, save_path, cut_size, cut_index_x, cut_index_y, patch_weight_num, patch_hegiht_num)
        self.so.progress_cut_update.emit(int((index / len(os.listdir(data_path))) * 100))
        self.so.splice_print.emit(self.data_preprocess.SpliceTextEdit, 'Finished {} cut'.format(index))
        
    
    def run_cutting(self, index):
        self.data_preprocess.startCut.setEnabled(False)
        self.data_preprocess.stopCut.setEnabled(True)
        root = self.data_preprocess.saveline.text()
        cut_size = int(self.params['Network_input_x_y'])
        channel = self.data_preprocess.buttonGroupCutting.checkedButton().text()
        cut_index_x = int(self.params['Cut_index_x'])
        cut_index_y = int(self.params['Cut_index_y'])
        patch_weight_num = int(self.params['Patch_weight_num'])
        patch_hegiht_num = int(self.params['Patch_height_num'])
        self.Cutting(root, cut_size, channel, cut_index_x, cut_index_y, patch_weight_num, patch_hegiht_num, index)

    def cut_workthread(self):
        root = os.path.join(self.data_preprocess.saveline.text(), 'brain_image_64_' + self.data_preprocess.buttonGroupCutting.checkedButton().text())
        data_list = [k for k in range(1, len(os.listdir(root)) + 1)]
        with ThreadPoolExecutor(max_workers=10) as pool:
            pool.map(self.run_cutting, data_list)
        self.so.progress_cut_update.emit(100)
        self.data_preprocess.startCut.setEnabled(True)
        self.data_preprocess.stopCut.setText('Finished')

    def Run(self):

        if len(self.data_preprocess.dataline.text()) == 0:
            QMessageBox.critical(
                self.data_preprocess,
                "Error",
                "Please select data path"
            )
        elif len(self.data_preprocess.saveline.text()) == 0:
            QMessageBox.critical(
                self.data_preprocess,
                "Error",
                "Please select data save path"
            )
        elif self.data_preprocess.radioButtonCut561.isChecked() == False and self.data_preprocess.radioButtonCut488.isChecked() == False and self.data_preprocess.radioButtonCut405.isChecked() == False:
            QMessageBox.critical(
                self.data_preprocess,
                "Warning",
                "Please select channel"
            )

        thread = Thread(target=self.cut_workthread)
        thread.start()

    


            