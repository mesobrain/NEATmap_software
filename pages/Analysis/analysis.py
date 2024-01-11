from Environment_ui import *
from PySide2.QtWidgets import QWidget, QFileDialog, QTextBrowser, QPushButton, QProgressBar, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import Signal, QObject
from threading import Thread
from pages.Analysis.seg3d_to_2d import seg3d_to_2d
from math import ceil,floor
from skimage import measure

class SignalStore(QObject):

    progress_update = Signal(QProgressBar, int)
    print_text = Signal(QTextBrowser, str)
    stop = Signal(QPushButton)

class CellCount(QWidget):
    def __init__(self, params) -> None:
        super(CellCount, self).__init__()
        self.signal = SignalStore()
        self.signal.progress_update.connect(self.UpdateProgress)
        self.signal.print_text.connect(self.Print)
        self.signal.stop.connect(self.CheckStop)

        self.cellcount = QUiLoader().load('pages/Analysis/analysis.ui')
        self.cellcount.Stop2dto3d.setEnabled(False)
        self.cellcount.StopCellCount.setEnabled(False)
        self.cellcount.StopExport.setEnabled(False)
        self.cellcount.DataRootLine.setPlaceholderText("Select BrainImgae/4.0 for the reconstruction data path")
        self.cellcount.PostPathLine.setPlaceholderText("Select posted whole brain data path")
        self.cellcount.SaveRootLine.setPlaceholderText("Select the path to save the data")

        self.cellcount.LoadData.clicked.connect(self.LoadData)
        self.cellcount.LoadPost.clicked.connect(self.LoadPost)
        self.cellcount.SeletSave.clicked.connect(self.SeletSave)

        self.cellcount.Start2dto3d.clicked.connect(self.run_segmentation_slice)
        self.cellcount.StartCellCount.clicked.connect(self.run_BrainImage2Spot)
        self.cellcount.StartExport.clicked.connect(self.run_Spot_csv)
        self.cellcount.Stop2dto3d.clicked.connect(self.Stopping2dto3d)
        self.cellcount.StopCellCount.clicked.connect(self.Stoppingcellcount)
        self.cellcount.StopExport.clicked.connect(self.Stoppingexport)

        self.cellcount.comboBox561.addItems(['C1', 'C2', 'C3', 'C4'])
        self.cellcount.comboBox488.addItems(['C1', 'C2', 'C3', 'C4'])
        self.cellcount.comboBox405.addItems(['C1', 'C2', 'C3', 'C4'])

        self.params = params

    def Print(self, fb, text):
        fb.appendPlainText(text)
    
    def UpdateProgress(self, fb, value):
        fb.setValue(value)

    def Stopping2dto3d(self): 
        self.cellcount.Stop2dto3d.setText('Stopping')
    
    def Stoppingcellcount(self): 
        self.cellcount.StopCellCount.setText('Stopping')

    def Stoppingexport(self): 
        self.cellcount.StopExport.setText('Stopping')

    def CheckStop(self, sp):
        if sp.text() == 'Stopping':
            os._exit(0)

    def LoadData(self):
        filepath = QFileDialog.getExistingDirectory(self.cellcount, 'Selet data root')
        self.cellcount.DataRootLine.setText(filepath)

    def LoadPost(self):
        filepath = QFileDialog.getExistingDirectory(self.cellcount, 'Selet post path')
        self.cellcount.PostPathLine.setText(filepath)

    def SeletSave(self):
        filepath = QFileDialog.getExistingDirectory(self.cellcount, 'Selet save path')
        self.cellcount.SaveRootLine.setText(filepath)

    def run_segmentation_slice(self):
        if len(self.cellcount.DataRootLine.text()) == 0:
            QMessageBox.critical(
                self.cellcount,
                "Error",
                "Please select raw data path."
            )
        elif len(self.cellcount.PostPathLine.text()) == 0:
            QMessageBox.critical(
                self.cellcount,
                "Error",
                "Please select posted whole brain data path."
            )
        elif len(self.cellcount.SaveRootLine.text()) == 0:
            QMessageBox.critical(
                self.cellcount,
                "Error",
                "Please select save path."
            )
        
        post_path = self.cellcount.PostPathLine.text()
        data_path = os.path.join(post_path, 'whole_brain_pred_post_filter')

        thread = Thread(target=self.segmentation_slice, args=(data_path, post_path, int(self.params['Thickness'])))
        thread.start()

    def segmentation_slice(self, data_path, post_path, slices, flip=False):
        self.cellcount.Start2dto3d.setEnabled(False)
        self.cellcount.Stop2dto3d.setEnabled(True)
        with tempfile.TemporaryDirectory() as TEMP_PATH:
            save_path = os.path.join(TEMP_PATH, 'whole_brain_pred_2d')
            seg3d_to_2d(data_path, save_path)
            save_root = os.path.join(post_path, 'whole_brain_pred_3d')
            BrainImage_root = save_path
            x = int(self.params['Brain_height'])
            y = int(self.params['Brain_weight'])
            if not os.path.exists(save_root):
                os.mkdir(save_root)
            brainimage_list =  os.listdir(BrainImage_root)
            brainimage_list.sort()
            for i in range(floor(len(brainimage_list)/slices)):
                self.signal.stop.emit(self.cellcount.Stop2dto3d)
                self.signal.progress_update.emit(self.cellcount.progressBar, int((i+1) / floor(len(brainimage_list)/slices) * 100))
                slice_list = brainimage_list[slices*i:slices*(i+1)]
                slice_image = np.zeros((slices,x,y)).astype('uint16')
                for j in range(len(slice_list)):
                    self.signal.print_text.emit(self.cellcount.SlicePrintText, 'Finished {} slice'.format(j))
                    im = tifffile.imread(os.path.join(BrainImage_root,slice_list[j]))
                    if flip:
                        im = np.fliplr(im)
                    slice_image[j,: ,:] = im
                Save_path = os.path.join(save_root,str(i).zfill(5)+'.tif')
                tifffile.imwrite(Save_path,slice_image, compress=2)
        self.cellcount.Stop2dto3d.setText('Finished')
        self.cellcount.Start2dto3d.setEnabled(True)

    def run_BrainImage2Spot(self):

        if len(self.cellcount.DataRootLine.text()) == 0:
            QMessageBox.critical(
                self.cellcount,
                "Error",
                "Please select raw data path."
            )
        elif len(self.cellcount.PostPathLine.text()) == 0:
            QMessageBox.critical(
                self.cellcount,
                "Error",
                "Please select posted whole brain data path."
            )
        elif len(self.cellcount.SaveRootLine.text()) == 0:
            QMessageBox.critical(
                self.cellcount,
                "Error",
                "Please select save path."
            )

        BrainImage_root = os.path.join(self.cellcount.SaveRootLine.text(), 'whole_brain_pred_3d')
        csv_root = os.path.join(self.cellcount.SaveRootLine.text(), 'whole_brain_cell_counts')
        thread = Thread(target=self.BrainImage2Spot, args=(BrainImage_root, csv_root))
        thread.start()

    def BrainImage2Spot(self, BrainImage_root,csv_root1):
        if not os.path.exists(csv_root1):
            os.mkdir(csv_root1)
        brainimage_list = os.listdir(BrainImage_root)
        brainimage_list.sort()
        save_file = open(os.path.join(csv_root1,'total.txt'),'w+')
        self.cellcount.StartCellCount.setEnabled(False)
        self.cellcount.StopCellCount.setEnabled(True)
        save_file.writelines('Id X Y Z Area\n')
        for i in range(0,len(brainimage_list)):
            self.signal.stop.emit(self.cellcount.StopCellCount)
            self.signal.progress_update.emit(self.cellcount.progressBar2, int((i+1) / len(brainimage_list) * 100))
            brainimage_path = os.path.join(BrainImage_root,brainimage_list[i])
            self.signal.print_text.emit(self.cellcount.CellCountPrintText, brainimage_path)
            self.signal.print_text.emit(self.cellcount.CellCountPrintText, 'image reading...')
            binary_image = tifffile.imread(brainimage_path)
            labeled_img = measure.label(binary_image, connectivity=1)
            properties = measure.regionprops(labeled_img)
            centroid_list = []
            area_list = []
            self.signal.print_text.emit(self.cellcount.CellCountPrintText, 'cell counting...')
            for pro in properties:
                centroid = pro.centroid
                centroid_list.append(centroid)
                area = pro.area
                area_list.append(area)
            centroid_list.sort()
            for j in range(len(centroid_list)):
                z = ceil(centroid_list[j][0])
                y = ceil(centroid_list[j][1])
                x = ceil(centroid_list[j][2])
                area = area_list[j]
                if area == int(self.params['filter_area_lower']) or area >= int(self.params['filter_area_upper']):
                    pass
                else:
                    z_index = z + i*int(self.params['Thickness'])
                    self.signal.print_text.emit(self.cellcount.CellCountPrintText, str(x) + ' ' + str(y) + ' ' + str(z_index) + ' ' + str(area) + '---' + str(j))
                    content = str(j) + ' ' + str(x) + ' '+ str(y) +' ' + str(z_index) +' ' + str(area) + '\n'
                    save_file.writelines(content)
        save_file.close()
        self.cellcount.StopCellCount.setText('Finished')
        self.cellcount.StartCellCount.setEnabled(True)

    def run_Spot_csv(self):

        if len(self.cellcount.DataRootLine.text()) == 0:
            QMessageBox.critical(
                self.cellcount,
                "Error",
                "Please select raw data path."
            )
        elif len(self.cellcount.PostPathLine.text()) == 0:
            QMessageBox.critical(
                self.cellcount,
                "Error",
                "Please select posted whole brain data path."
            )
        elif len(self.cellcount.SaveRootLine.text()) == 0:
            QMessageBox.critical(
                self.cellcount,
                "Error",
                "Please select save path."
            )

        csv_root = os.path.join(self.cellcount.SaveRootLine.text(), 'whole_brain_cell_counts')
        save_csv_root = os.path.join(csv_root, 'Thumbnail_CSV')
        total_path = os.path.join(csv_root, 'total.txt')
        json_path = os.path.join(self.cellcount.DataRootLine.text(), '..', 'freesia_4.0_'+ self.cellcount.comboBox405.currentText() + '_405nm_10X.json')
        with open(json_path) as f:
            brain = json.load(f)
            images = brain['images']
            total_num = len(images)
        thread = Thread(target=self.Spot_csv, args=(total_path, save_csv_root, total_num, float(self.params['group_num'])))
        thread.start()

    def Spot_csv(self, total_path, csv_root, brainimage2d_num, group_num):
        if not os.path.exists(csv_root):
            os.mkdir(csv_root)
        f = open(total_path,'r')
        self.cellcount.StartExport.setEnabled(False)
        self.cellcount.StopExport.setEnabled(True)
        spots = []
        for spot in f.readlines()[1:]:
            a = spot.strip().split(' ')
            a = np.array(a).astype(dtype=int).tolist()
            spots.append(a)
        # print(spots)

        # group with z = spots[3] , for every 6.25 slices
        for i in range(floor(brainimage2d_num/group_num)):
            self.signal.stop.emit(self.cellcount.StopExport)
            self.signal.progress_update.emit(self.cellcount.progressBar3, int((i+1) / floor(brainimage2d_num/group_num) * 100))
            self.signal.print_text.emit(self.cellcount.ExportPrintText, '----Thumbnail'+str(i))
            csv_name = str(i).zfill(4)+'_25.0.tif.csv'
            csv_path = os.path.join(csv_root,csv_name)
            count = 0
            list_name = ['Position X','Position Y','Position Z', 'Unit', 'Category', 'Collection','Time','ID']
            with open(csv_path,'w+',newline='')as f:
                csv_write = csv.writer(f, dialect='excel')
                csv_write.writerow(['25.0'])
                csv_write.writerow(['=========='])
                csv_write.writerow(list_name)
                for j in range(len(spots)):
                    if i * group_num<=spots[j][3]<=i * group_num + (group_num-1):
                        self.signal.print_text.emit(self.cellcount.ExportPrintText, str(spots[j]))
                        x ,y ,z = str(spots[j][1]*int(self.params['voxel_size'])), str(spots[j][2]*int(self.params['voxel_size'])), str(spots[j][3]*int(self.params['voxel_size']))
                        writeline = [x,y,z,'um','Spot','Position','1',str(count)]
                        count += 1
                        csv_write.writerow(writeline)
                        self.signal.print_text.emit(self.cellcount.ExportPrintText, str(count))
        self.cellcount.StopExport.setText('Finished')
        self.cellcount.StartExport.setEnabled(True)
        