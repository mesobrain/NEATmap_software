from Environment_ui import *
from PySide2.QtWidgets import QWidget, QFileDialog, QTextBrowser, QPushButton, QProgressBar, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import Signal, QObject
from threading import Thread
from pages.Splice.restore import create_residual_image, concat, load

class SignalStore(QObject):

    progress_update = Signal(QProgressBar, int)
    splice_print = Signal(QTextBrowser, str)
    stop_splice = Signal(QPushButton)

class Splice(QWidget):

    def __init__(self, params):
        super(Splice, self).__init__()
        self.signal = SignalStore()
        self.signal.progress_update.connect(self.UpdateProgress)
        self.signal.splice_print.connect(self.Print)
        self.signal.stop_splice.connect(self.CheckStop)

        self.splice = QUiLoader().load('pages/Splice/splice.ui')

        self.splice.Stop.setEnabled(False)
        self.splice.DataRootLine.setPlaceholderText("Select BrainImgae/4.0 for the reconstruction data path")
        self.splice.InferRootLine.setPlaceholderText("Select the segmented whole brain data path")
        self.splice.BrainRootLine.setPlaceholderText("Select brain patch path")
        self.splice.SaveRootLine.setPlaceholderText("Select the path to save the data")

        self.splice.LoadData.clicked.connect(self.LoadData)
        self.splice.LoadInfer.clicked.connect(self.LoadInfer)
        self.splice.LoadBrain.clicked.connect(self.LoadBrain)
        self.splice.Selet.clicked.connect(self.SeletSaveRoot)
        self.splice.Start.clicked.connect(self.run_splice)
        self.splice.Stop.clicked.connect(self.StopSeg)

        self.splice.comboBox561.addItems(['C1', 'C2', 'C3', 'C4'])
        self.splice.comboBox488.addItems(['C1', 'C2', 'C3', 'C4'])

        self.params = params

    def Print(self, fb, text):
        fb.appendPlainText(text)
    
    def UpdateProgress(self, fb, value):
        fb.setValue(value)

    def StopSeg(self): 
        self.splice.Stop.setText('Stopping')

    def CheckStop(self, sp):
        if sp.text() == 'Stopping':
            os._exit(0)

    def LoadData(self):
        filepath = QFileDialog.getExistingDirectory(self.splice, 'Selet data root')
        self.splice.DataRootLine.setText(filepath)

    def LoadInfer(self):
        filepath = QFileDialog.getExistingDirectory(self.splice, 'Selet infer path')
        self.splice.InferRootLine.setText(filepath)

    def LoadBrain(self):
        filepath = QFileDialog.getExistingDirectory(self.splice, 'Selet brain path')
        self.splice.BrainRootLine.setText(filepath)

    def SeletSaveRoot(self):
        filepath = QFileDialog.getExistingDirectory(self.splice, 'Selet save path')
        self.splice.SaveRootLine.setText(filepath)

    def run_splice(self):

        if len(self.splice.DataRootLine.text()) == 0:
            QMessageBox.critical(
                self.splice,
                "Error",
                "Please select data path."
            )
        elif len(self.splice.InferRootLine.text()) == 0:
            QMessageBox.critical(
                self.splice,
                "Error",
                "Please select segmented whole brain data path."
            )
        elif len(self.splice.BrainRootLine.text()) == 0:
            QMessageBox.critical(
                self.splice,
                "Error",
                "Please select brain patch path."
            )
        elif len(self.splice.SaveRootLine.text()) == 0:
            QMessageBox.critical(
                self.splice,
                "Error",
                "Please select save path."
            )
        elif self.splice.radioButton561.isChecked() == False and self.splice.radioButton488.isChecked() == False:
            QMessageBox.critical(
                self.splice,
                "Error",
                "Please select channel."
            )

        json_path = os.path.join(self.splice.DataRootLine.text(), '..', 'freesia_4.0_'+ self.splice.comboBox488.currentText() + '_488nm_10X.json')
        with open(json_path) as f:
            brain = json.load(f)
            images = brain['images']
            total_num = len(images)

        channel = self.splice.buttonGroup.checkedButton().text()
        pred_root = os.path.join(self.splice.InferRootLine.text(), 'whole_predications_' + channel)

        thread = Thread(target=self.SpliceSegPatch, args=(total_num, pred_root, channel))
        self.splice.Start.setEnabled(False)
        self.splice.Stop.setEnabled(True)
        thread.start()

    def SpliceSegPatch(self, total_num, pred_root, channel):

        brain_path = os.path.join(self.splice.BrainRootLine.text(), 'brain_image_64_' + channel)
        save_path_3d = os.path.join(self.splice.SaveRootLine.text(), 'whole_brain_pred_' + channel)
        os.makedirs(save_path_3d, exist_ok=True)
        infer_total_num = len(os.listdir(brain_path))
        resuidual_z = total_num - (int(self.params['Network_input_z']) * infer_total_num)
        patch_total_num = int(self.params['Patch_weight_num']) * int(self.params['Patch_height_num'])

        for num in range(1, len(os.listdir(brain_path)) + 1):
            self.signal.stop_splice.emit(self.splice.Stop)
            self.signal.progress_update.emit(self.splice.progressBar, int(num / len(os.listdir(brain_path)) * 100))
            split_path = pred_root + '/brain_predications_{}_swin_epoch10'.format(num) + '/VISoR256/' + self.params['checkpoint_name']
            list = load(split_path, num, total_patch_num=patch_total_num)
            concat(list, save_path_3d, num, param=self.params)
            self.signal.splice_print.emit(self.splice.SplicePrintText, 'finished {} image'.format(num))
        create_residual_image(save_path_3d, infer_total_num + 1, resuidual_z, param=self.params)

        self.splice.Stop.setText('Finished')
        self.splice.Start.setEnabled(True)
        
