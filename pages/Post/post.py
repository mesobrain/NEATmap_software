from tkinter.messagebox import NO
from Environment_ui import *
from PySide2.QtWidgets import QWidget, QFileDialog, QTextBrowser, QPushButton, QProgressBar, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import Signal, QObject
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from pages.Splice.utils import segmentation_quick_view
from pages.Post.postprocessing import big_object_filter, post_488nm, remove_piont
from pages.Post.intensity import get_2d_intensity
from scipy import ndimage as ndi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class SignalStore(QObject):

    progress_update = Signal(QProgressBar, int)
    post_print = Signal(QTextBrowser, str)
    stop_post = Signal(QPushButton)

class PostProcess(QWidget):

    def __init__(self, params):
        super(PostProcess, self).__init__()
        self.signal = SignalStore()
        self.signal.progress_update.connect(self.UpdateProgress)
        self.signal.post_print.connect(self.PostPrint)
        self.signal.stop_post.connect(self.CheckStop)

        self.post_process = QUiLoader().load('pages/Post/post.ui')
        self.post_process.Stop.setEnabled(False)
        self.post_process.Stop_intensity.setEnabled(False)
        self.post_process.BrainRootLine.setPlaceholderText("Select brain patch path")
        self.post_process.SplicePathLine.setPlaceholderText("Select the spliced whole brain data path")
        self.post_process.SaveRootLine.setPlaceholderText("Select the path to save the data")

        self.post_process.LoadBrain.clicked.connect(self.LoadBrain)
        self.post_process.LoadSpliced.clicked.connect(self.LoadSplice)
        self.post_process.SeletSave.clicked.connect(self.SeletSave)
        self.post_process.Start.clicked.connect(self.Run)
        self.post_process.Stop.clicked.connect(self.StopPost)
        self.post_process.Start_intensity.clicked.connect(self.Run_intensity)
        self.post_process.Stop_intensity.clicked.connect(self.StopPost)
        self.post_process.Export_intensity.clicked.connect(self.update_plot_csv)

        self.plot = QUiLoader().load('pages/Post/plot.ui')
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.plot.verticalLayout.addWidget(self.canvas)

        self.params = params

    def PostPrint(self, fb, text):
        fb.appendPlainText(text)
    
    def UpdateProgress(self, fb, value):
        fb.setValue(value)

    def StopPost(self): 
        self.post_process.Stop.setText('Stopping')

    def CheckStop(self, sp):
        if sp.text() == 'Stopping':
            os._exit(0)

    def LoadBrain(self):
        filepath = QFileDialog.getExistingDirectory(self.post_process, 'Selet brain path')
        self.post_process.BrainRootLine.setText(filepath)

    def LoadSplice(self):
        filepath = QFileDialog.getExistingDirectory(self.post_process, 'Selet splice path')
        self.post_process.SplicePathLine.setText(filepath)

    def SeletSave(self):
        filepath = QFileDialog.getExistingDirectory(self.post_process, 'Selet save root')
        self.post_process.SaveRootLine.setText(filepath)

    def run_post(self, index):
        self.post_process.Start.setEnabled(False)
        image_path = self.post_process.BrainRootLine.text()
        spliced_path = self.post_process.SplicePathLine.text()
        pred_path = os.path.join(spliced_path, 'whole_brain_pred_' + self.params['Staining channel'])
        if os.path.exists(os.path.join(spliced_path, 'whole_brain_pred_' + self.params['Autofluo channel'])):
            path_488nm = os.path.join(spliced_path, 'whole_brain_pred_' + self.params['Autofluo channel'])
        else:
            path_488nm = None
        save_path = os.path.join(self.post_process.SaveRootLine.text(), 'whole_brain_pred_post_filter')
        min_size = int(self.params['point_min_size']) 
        max_intensity = int(self.params['big_object_size'])
        lower_limit = int(self.params['intensity_lower_differ']) if self.params['intensity_lower_differ'] else None
        upper_limit = int(self.params['intensity_upper_differ']) if self.params['intensity_upper_differ'] else None
        self.spot_filter(image_path, pred_path, path_488nm, save_path, min_size, 
                        max_intensity, index, lower_limit=lower_limit, upper_limit=upper_limit)

    def spot_filter(self, image_path, pred_path, path_488nm, save_path, min_size, max_intensity, index, lower_limit=None, upper_limit=None):
        os.makedirs(save_path, exist_ok=True)

        self.signal.stop_post.emit(self.post_process.Stop)
        if index < len(os.listdir(pred_path)):
            img = tifffile.imread(os.path.join(image_path, 'Z{:05d}.tif'.format(index)))
            seg = tifffile.imread(os.path.join(pred_path, 'Z{:05d}_seg.tif'.format(index)))
            if path_488nm is not None:
                seg_488nm = tifffile.imread(os.path.join(path_488nm, 'Z{:05d}_seg.tif'.format(index)))
                seg = segmentation_quick_view(seg)
                seg_488nm = segmentation_quick_view(seg_488nm)
                post_seg = post_488nm(seg, seg_488nm)
            else:
                post_seg = segmentation_quick_view(seg)
            resMatrix = remove_piont(post_seg>0, min_size)
            resMatrix = segmentation_quick_view(resMatrix)
            filter_matrix = img.astype(np.float32) - resMatrix.astype(np.float32)
            if lower_limit is not None:                
                bool_matrix = np.logical_and(filter_matrix >= lower_limit - 255, filter_matrix <= upper_limit - 255)
                index_matrix = np.where(bool_matrix == False)

                for j in range(len(index_matrix[0])):
                    resMatrix[index_matrix[0][j], index_matrix[1][j], index_matrix[2][j]] = 0
            
            new_seg = big_object_filter(filter_matrix, resMatrix, limit=max_intensity)
        else:
            new_seg = tifffile.imread(os.path.join(pred_path, 'Z{:05d}_seg.tif'.format(index)))
        tifffile.imwrite(os.path.join(save_path, 'Z{:05d}_filter.tif'.format(index)), new_seg.astype('uint16'))
        self.signal.post_print.emit(self.post_process.PostPrintText, 'Finished {} segmentation filter'.format(index))
        self.signal.progress_update.emit(self.post_process.progressBar, int(index / len(os.listdir(pred_path)) * 100))    

    def post_workthread(self):
        spliced_path = self.post_process.SplicePathLine.text()
        pred_path = os.path.join(spliced_path, 'whole_brain_pred_' + self.params['Staining channel'])
        index = [k for k in range(1, len(os.listdir(pred_path)) + 1)]
        self.post_process.Start.setEnabled(False)
        self.post_process.Stop.setEnabled(True)
        with ThreadPoolExecutor(max_workers=10) as pool:
            pool.map(self.run_post, index)
        self.signal.progress_update.emit(self.post_process.progressBar, 100)
        self.post_process.Stop.setText('Finished')
        self.post_process.Start.setEnabled(True)
    
    def Run(self):
        if len(self.post_process.BrainRootLine.text()) == 0:
            QMessageBox.critical(
                self.post_process,
                "Error",
                "Please select brain patch data path."
            )
        elif len(self.post_process.SplicePathLine.text()) == 0:
            QMessageBox.critical(
                self.post_process,
                "Error",
                "Please select spliced whole brain data path."
            )
        elif len(self.post_process.SaveRootLine.text()) == 0:
            QMessageBox.critical(
                self.post_process,
                "Error",
                "Please select save path."
            )
        thread = Thread(target=self.post_workthread)
        thread.start()

    def run_stats_intensity(self, index):
        
        image_path = self.post_process.BrainRootLine.text()
        post_path = os.path.join(self.post_process.SaveRootLine.text(), 'whole_brain_pred_post_filter')
        save_path = os.path.join(self.post_process.SaveRootLine.text(), 'whole_brain_intensity')
        self.stats_intensity(image_path, post_path, save_path, index)

    def stats_intensity(self, data_path, seg_path, save_path, index, min_volume=2, max_volume=64):
        os.makedirs(save_path, exist_ok=True)
        image = tifffile.imread(os.path.join(data_path, 'Z{:05d}.tif'.format(index)))
        seg = tifffile.imread(os.path.join(seg_path, 'Z{:05d}_filter.tif'.format(index)))
        raw = image[image.shape[0] // 2 - 1, :, :].copy()
        label = seg[image.shape[0] // 2 - 1, :, :].copy()
        struct = ndi.generate_binary_structure(2, 1)
        label, _ = ndi.label(label, struct)
        unique, counts = np.unique(label, return_counts=True)
        small, medium, large = [], [], []
        for uq, ct in zip(unique, counts):
            if uq == 0:
                continue # skip zero!
            if ct <= min_volume:
                small.append([uq, ct]) # if object is smaller than mimimum size, it gets excluded
            elif min_volume < ct <= max_volume:
                medium.append([uq, ct] )
            else:
                large.append([uq, ct])
            detected_object = []
        object_ids = [e[0] for e in medium]
        volumes = [e[1] for e in medium]
        if object_ids: # skip if empty
            center_mass = ndi.center_of_mass(raw, label, object_ids )
            coms = np.array(center_mass).round().astype(np.int)
            for i, com in enumerate(coms):
                this_idx = object_ids[i]
                deltaI, bg = get_2d_intensity(raw, label, this_idx, com, mode='obj_mean')
                vol = volumes[i]
                obj = [com[1], com[0], bg, bg + deltaI, vol] # X, Y, bg, intensity, volume
                detected_object.append(obj)
        csv_name = 'intensity_{}.csv'.format(index)
        csv_path = os.path.join(save_path, csv_name)
        list_name = ['X', 'Y', 'bg_percentile', 'intensity', 'volume']
        with open(csv_path,'w+',newline='')as f:
            csv_write = csv.writer(f, dialect='excel')
            csv_write.writerow(list_name)
            for k in range(len(detected_object)):
                csv_write.writerow(detected_object[k])

        self.signal.progress_update.emit(self.post_process.progressBar_inetensity, int(index / len(os.listdir(data_path)) * 100))
        self.signal.post_print.emit(self.post_process.PostPrintText, 'Finished {} csv'.format(index))

    def intensity_workthread(self):
        index = [k for k in range(1, len(os.listdir(self.post_process.BrainRootLine.text())) + 1)]
        self.post_process.Start_intensity.setEnabled(False)
        self.post_process.Stop_intensity.setEnabled(True)
        with ThreadPoolExecutor(max_workers=10) as pool:
            pool.map(self.run_stats_intensity, index)
        self.signal.progress_update.emit(self.post_process.progressBar_inetensity, 100)
        self.post_process.Stop_intensity.setText('Finished')
        self.post_process.Start_intensity.setEnabled(True)

    def Run_intensity(self):
        if len(self.post_process.BrainRootLine.text()) == 0:
            QMessageBox.critical(
                self.post_process,
                "Error",
                "Please select brain patch data path."
            )
        elif len(self.post_process.SplicePathLine.text()) == 0:
            QMessageBox.critical(
                self.post_process,
                "Error",
                "Please select spliced whole brain data path."
            )
        elif len(self.post_process.SaveRootLine.text()) == 0:
            QMessageBox.critical(
                self.post_process,
                "Error",
                "Please select save path."
            )
        thread = Thread(target=self.intensity_workthread)
        thread.start()

    def update_plot_csv(self):
        self.post_process.Export_intensity.setEnabled(False)
        intensity_root = os.path.join(self.post_process.SaveRootLine.text(), 'whole_brain_intensity')
        save_path = os.path.join(self.post_process.SaveRootLine.text(), 'Export_file')
        os.makedirs(save_path, exist_ok=True)
        intensity_values = []
        for i in range(1, len(os.listdir(intensity_root)) + 1):
            csv_path = os.path.join(intensity_root, 'intensity_{}.csv'.format(i))
            file = open(csv_path)
            df = pd.read_csv(file)
            intensity = df['intensity'].values
            intensity_values.extend(intensity)
        j = 0    
        while j < len(intensity_values):
            if intensity_values[j] > int(self.params['intensity_upper_differ']):
                del intensity_values[j]
            else:
                j += 1

        intensity_range = (min(intensity_values), max(intensity_values))
        hist, bin_edges = np.histogram(intensity_values, bins=20,
                                        range=intensity_range,
                                        density=False)
        bins = ( bin_edges[:-1] + bin_edges[1:] ) / 2
        savename = os.path.join(save_path, 'whole_brain_detected_c-Fos_signal_intensity.csv')
        values = pd.DataFrame(columns=['intensity', 'normalized_count'])
        values['intensity'] = bins
        values['normalized_count'] = hist
        values.to_csv(savename, index=False)
        self.signal.post_print.emit(self.post_process.PostPrintText, 'Export csv file')

        self.signal.post_print.emit(self.post_process.PostPrintText, 'Drawing histigram ...')
        self.ax.bar(bins, hist, width=10, align='center', alpha=0.7, color='b')
        self.ax.set_xlabel('Intensity')
        self.ax.set_ylabel('Counts')
        self.ax.set_title('Whole brain detected c-Fos positive signal intensity')
        self.figure.savefig(os.path.join(save_path, 'Intensity.png'))
        self.canvas.draw()
        self.plot.show()
        self.post_process.Export_intensity.setEnabled(True)