from Environment_ui import *
from PySide2.QtWidgets import QWidget, QFileDialog, QPushButton, QLabel, QProgressBar
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import Signal, QObject
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader
from tqdm import tqdm
from math import ceil,floor
from skimage import measure

from pages.Data_preprocessing.cutting import single_cutting
from pages.Whole_brain_seg.utils import test_each_brain
from pages.Whole_brain_seg.dataset_whole_brain import Whole_brain_dataset
from pages.Whole_brain_seg.swin_transform import swin_tiny_patch4_window8_256
from pages.Splice.restore import create_residual_image, concat, load
from pages.Post.postprocessing import big_object_filter, post_488nm, remove_piont
from pages.Splice.utils import segmentation_quick_view
from pages.Registration.brain_registration import read_freesia2_image, write_freesia2_image
from pages.Registration.elastix_files import get_align_transform
from pages.Analysis.seg3d_to_2d import seg3d_to_2d

ROOT_DIR = 'pages/Registration'
PARAMETER_DIR = 'pages/Registration/parameters'

class WorkerThread(Thread):
    def __init__(self, work_queue, worker_function):
        super().__init__()
        self.work_queue = work_queue
        self.worker_function_queue = worker_function

    def run(self):
        while True:
            task = self.work_queue.get()
            worker_function = self.worker_function_queue.get()
            if task is None:
                break
            worker_function(*task)

class SignalStore(QObject):
    progress_update = Signal(QProgressBar, int)
    state = Signal(QLabel)
    stop_pipeline = Signal(QPushButton)

class NEATmapPipeline(QWidget):
    def __init__(self, params):
        super(NEATmapPipeline, self).__init__()
        self.signal = SignalStore()
        self.signal.progress_update.connect(self.UpdateProgress)
        self.signal.stop_pipeline.connect(self.CheckStop)
        self.signal.state.connect(self.ChangeState)

        self.neatmap_pipeline = QUiLoader().load('pages/User_friendly/user_friendly_ui.ui')

        self.neatmap_pipeline.DataRoot_lineEdit.setPlaceholderText("Select BrainImgae/4.0 for the reconstruction data path")
        self.neatmap_pipeline.SaveRoot_lineEdit.setPlaceholderText("Select the path to save the data")
        self.neatmap_pipeline.SnapshotPath_lineEdit.setPlaceholderText('Select the weight parameters (pth file) after training.')

        self.neatmap_pipeline.comboBox561.addItems(['C1', 'C2', 'C3', 'C4'])
        self.neatmap_pipeline.comboBox488.addItems(['C1', 'C2', 'C3', 'C4'])
        self.neatmap_pipeline.comboBox405.addItems(['C1', 'C2', 'C3', 'C4'])   

        self.neatmap_pipeline.Data_progressBar.setMaximum(100)    
        self.neatmap_pipeline.Cut_progressBar.setMaximum(100)
        self.neatmap_pipeline.Segmenter_progressBar.setMaximum(100)
        self.neatmap_pipeline.Splice_progressBar.setMaximum(100)
        self.neatmap_pipeline.Post_progressBar.setMaximum(100)
        self.neatmap_pipeline.BrainSlice_progressBar.setMaximum(100)
        self.neatmap_pipeline.CellCounts_progressBar.setMaximum(100)
        self.neatmap_pipeline.ExportTable_progressBar.setMaximum(100)

        self.neatmap_pipeline.DataRootLoad.clicked.connect(self.SeletData)
        self.neatmap_pipeline.SaveRootLoad.clicked.connect(self.SeletSave)
        self.neatmap_pipeline.SnapshotPathLoad.clicked.connect(self.SeletSnapshot)

        self.neatmap_pipeline.start.clicked.connect(self.Pipeline)

        self.neatmap_pipeline.stop.setEnabled(False)

        self.neatmap_pipeline.checkbox_group.setExclusive(True)

        parser = argparse.ArgumentParser()
        parser.add_argument('--volume_path', type=str,
                            default='', help='root dir for validation volume data') 
        parser.add_argument('--dataset', type=str,
                            default='VISoR', help='experiment_name')
        parser.add_argument('--max_epochs', type=int,
                            default=int(params['max_epochs']), help='maximum epoch number to train')
        parser.add_argument('--num_classes', type=int,
                            default=int(params['num_classes']), help='output channel of network')
        parser.add_argument('--list_dir', type=str,
                            default='./lists', help='list dir')
        parser.add_argument('--model_name', type=str,
                            default='swin', help='model_name')

        parser.add_argument('--batch_size', type=int, default=int(params['test_batch_size']), help='batch_size per gpu')
        parser.add_argument('--img_size', type=int, default=int(params['Network_input_x_y']), help='input patch size of network input')
        parser.add_argument('--channel', type=str, default=params['Staining channel'], help='selecting the channel for reasoning about the whole brain')
        parser.add_argument('--is_savetif', action="store_true", default=True, help='whether to save results during inference')
        parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as tif!')
        parser.add_argument('--base_lr', type=float,  default=float(params['lr']), help='segmentation network learning rate')
        parser.add_argument('--patch_csv_root', type=str,  default=params['save_patch_metrics'], help='The path to save the patch evaluation result csv')
        parser.add_argument('--slice_csv_root', type=str,  default=params['save_metrics'], help='The path to save the slice evaluation result csv')

        self.args = parser.parse_args(args=[])

        self.params = params

    def SeletData(self):
        filepath = QFileDialog.getExistingDirectory(self.neatmap_pipeline, 'Selet root')
        self.neatmap_pipeline.DataRoot_lineEdit.setText(filepath)
    
    def SeletSave(self):
        filepath = QFileDialog.getExistingDirectory(self.neatmap_pipeline, 'Selet root')
        self.neatmap_pipeline.SaveRoot_lineEdit.setText(filepath)

    def SeletSnapshot(self):
        filepath = QFileDialog.getExistingDirectory(self.neatmap_pipeline, 'Selet snapshot path')
        self.neatmap_pipeline.SnapshotPath_lineEdit.setText(filepath)

    def UpdateProgress(self, fb, value):
        fb.setValue(value)

    def CheckStop(self, sp):
        if sp.text() == 'Stopping':
            os._exit(0)

    def ChangeState(self, label):
        label.setText("Finished")

    def brain2dto3d(self, total_num, select_channel):
        z_num = int(self.params['Network_input_z'])
        name_index = 1
        temp = []
        j = 0
        save_path = self.neatmap_pipeline.SaveRoot_lineEdit.text() + '/brain_image_64_' + self.neatmap_pipeline.buttonGroup.checkedButton().text()
        os.makedirs(save_path, exist_ok=True)
        for i in range(0, total_num):
            self.signal.stop_pipeline.emit(self.neatmap_pipeline.stop)
            self.signal.progress_update.emit(self.neatmap_pipeline.Data_progressBar, int(((i+1) / total_num) * 100))
            image = sitk.ReadImage(os.path.join(self.neatmap_pipeline.DataRoot_lineEdit.text(), 'Z{:05d}_'.format(i) + select_channel +'.tif'))
            array = sitk.GetArrayFromImage(image)
            temp.append(array)
            if i == z_num - 1 + j:
                tif = np.array(temp)
                tif = sitk.GetImageFromArray(tif)
                sitk.WriteImage(tif, os.path.join(save_path, 'Z{:05d}.tif'.format(name_index)))
                j += z_num
                name_index += 1
                temp = []
        self.signal.state.emit(self.neatmap_pipeline.brain_image_state)

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

        self.signal.stop_pipeline.emit(self.neatmap_pipeline.stop)
        if cut_label:
            name = 'Z{:05d}_seg'.format(index)
        else:
            name = 'Z{:05d}'.format(index)
        image = os.path.join(data_path, name + '.tif')
        single_cutting(image, index, save_path, cut_size, cut_index_x, cut_index_y, patch_weight_num, patch_hegiht_num)
        self.signal.progress_update.emit(self.neatmap_pipeline.Cut_progressBar, int((index / len(os.listdir(data_path))) * 100))

    def run_cutting(self, index):

        root = self.neatmap_pipeline.SaveRoot_lineEdit.text()
        cut_size = int(self.params['Network_input_x_y'])
        channel = self.neatmap_pipeline.buttonGroup.checkedButton().text()
        cut_index_x = int(self.params['Cut_index_x'])
        cut_index_y = int(self.params['Cut_index_y'])
        patch_weight_num = int(self.params['Patch_weight_num'])
        patch_hegiht_num = int(self.params['Patch_height_num'])
        self.Cutting(root, cut_size, channel, cut_index_x, cut_index_y, patch_weight_num, patch_hegiht_num, index)

    def cut_workthread(self):
        root = os.path.join(self.neatmap_pipeline.SaveRoot_lineEdit.text(), 'brain_image_64_' + self.neatmap_pipeline.buttonGroup.checkedButton().text())
        data_list = [k for k in range(1, len(os.listdir(root)) + 1)]
        with ThreadPoolExecutor(max_workers=10) as pool:
            pool.map(self.run_cutting, data_list)
        self.signal.progress_update.emit(self.neatmap_pipeline.Cut_progressBar, 100)
        self.signal.state.emit(self.neatmap_pipeline.cutting_state)

    def infer_each_brain(self, args, model, ind, test_save_path=None):
        test_data = args.Dataset(base_dir=args.volume_path, split=args.name_list, list_dir=args.list_dir)
        testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model.eval()
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            image, case_name = sampled_batch["image"], sampled_batch['case_name'][0]
            test_each_brain(image, model, patch_size=[args.img_size, args.img_size], test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)

    def inference_whole_brain_seg(self, channel, patch_path, snapshot_path):
        test_path = os.path.join(patch_path, 'PatchImage_' + channel)
        for ind in range(1, len(os.listdir(test_path)) + 1):
            self.signal.stop_pipeline.emit(self.neatmap_pipeline.stop)
            self.signal.progress_update.emit(self.neatmap_pipeline.Segmenter_progressBar, int(ind / len(os.listdir(test_path)) * 100))
            dataset_config = {
                'VISoR': {
                    'Dataset': Whole_brain_dataset,
                    'volume_path': test_path + '/patchimage{}'.format(ind),
                    'list_dir': self.params['whole_brain_list'],
                    'name_list': 'Z{:05d}_test'.format(ind),
                    'z_spacing': int(self.params['z_spacing']),
                },
            }
            dataset_name = self.args.dataset
            self.args.exp = dataset_name + str(self.args.img_size)
            self.args.volume_path = dataset_config[dataset_name]['volume_path']
            self.args.Dataset = dataset_config[dataset_name]['Dataset']
            self.args.list_dir = dataset_config[dataset_name]['list_dir']
            self.args.name_list = dataset_config[dataset_name]['name_list']
            self.args.z_spacing = dataset_config[dataset_name]['z_spacing']
            self.args.is_pretrain = True

            net = swin_tiny_patch4_window8_256(in_channels=int(self.params['swin_in_channels']), num_classes=self.args.num_classes).cuda()

            snapshot = os.path.join(snapshot_path, 'best_model.pth')
            if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(self.args.max_epochs-1))
            net.load_state_dict(torch.load(snapshot))
            snapshot_name = snapshot_path.split('/')[-1]
            if self.args.is_savetif:
                self.args.test_save_dir = patch_path + '/whole_predications_' + channel + '/brain_predications_{}_swin_epoch10'.format(ind)
                test_save_path = os.path.join(self.args.test_save_dir, self.args.exp, snapshot_name)
                os.makedirs(test_save_path, exist_ok=True)
            else:
                test_save_path = None
            self.infer_each_brain(self.args, net, ind, test_save_path)
        self.signal.state.emit(self.neatmap_pipeline.segmenter_state)

    def SpliceSegPatch(self, total_num, pred_root, channel):

        brain_path = os.path.join(self.neatmap_pipeline.SaveRoot_lineEdit.text(), 'brain_image_64_' + channel)
        save_path_3d = os.path.join(self.neatmap_pipeline.SaveRoot_lineEdit.text(), 'whole_brain_pred_' + channel)
        os.makedirs(save_path_3d, exist_ok=True)
        infer_total_num = len(os.listdir(brain_path))
        resuidual_z = total_num - (int(self.params['Network_input_z']) * infer_total_num)
        patch_total_num = int(self.params['Patch_weight_num']) * int(self.params['Patch_height_num'])

        for num in range(1, len(os.listdir(brain_path)) + 1):
            self.signal.stop_pipeline.emit(self.neatmap_pipeline.stop)
            self.signal.progress_update.emit(self.neatmap_pipeline.Splice_progressBar, int(num / len(os.listdir(brain_path)) * 100))
            split_path = pred_root + '/brain_predications_{}_swin_epoch10'.format(num) + '/VISoR256/' + self.params['checkpoint_name']
            list = load(split_path, num, total_patch_num=patch_total_num)
            concat(list, save_path_3d, num, param=self.params)
        create_residual_image(save_path_3d, infer_total_num + 1, resuidual_z, param=self.params)
        self.signal.state.emit(self.neatmap_pipeline.splice_state)

    def run_post(self, index):
        image_path = os.path.join(self.neatmap_pipeline.SaveRoot_lineEdit.text(), 'brain_image_64_561nm')
        # autofluo_image_path = os.path.join(self.neatmap_pipeline.SaveRoot_lineEdit.text(), 'brain_image_64_488nm')
        spliced_path = self.neatmap_pipeline.SaveRoot_lineEdit.text()
        pred_path = os.path.join(spliced_path, 'whole_brain_pred_' + self.params['Staining channel'])
        if os.path.exists(os.path.join(spliced_path, 'whole_brain_pred_' + self.params['Autofluo channel'])):
            path_488nm = os.path.join(spliced_path, 'whole_brain_pred_' + self.params['Autofluo channel'])
        else:
            path_488nm = None
        save_path = os.path.join(self.neatmap_pipeline.SaveRoot_lineEdit.text(), 'whole_brain_pred_post_filter')
        min_size = int(self.params['point_min_size']) 
        max_intensity = int(self.params['big_object_size'])
        lower_limit = int(self.params['intensity_lower_differ']) if self.params['intensity_lower_differ'] else None
        upper_limit = int(self.params['intensity_upper_differ']) if self.params['intensity_upper_differ'] else None
        self.spot_filter(image_path, pred_path, path_488nm, save_path, min_size, 
                        max_intensity, index, lower_limit=lower_limit, upper_limit=upper_limit)

    def spot_filter(self, image_path, pred_path, path_488nm, save_path, min_size, max_intensity, index, lower_limit=None, upper_limit=None):
        os.makedirs(save_path, exist_ok=True)
        self.signal.stop_pipeline.emit(self.neatmap_pipeline.stop)
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
        self.signal.progress_update.emit(self.neatmap_pipeline.Post_progressBar, int(index / len(os.listdir(pred_path)) * 100))    

    def post_workthread(self):
        spliced_path = self.neatmap_pipeline.SaveRoot_lineEdit.text()
        pred_path = os.path.join(spliced_path, 'whole_brain_pred_' + self.params['Staining channel'])
        index = [k for k in range(1, len(os.listdir(pred_path)) + 1)]
        with ThreadPoolExecutor(max_workers=10) as pool:
            pool.map(self.run_post, index)
        self.signal.progress_update.emit(self.neatmap_pipeline.Post_progressBar, 100)
        self.signal.state.emit(self.neatmap_pipeline.post_state)

    def register_brain(self, image_list_file:str, output_path: str, template_file: str, output_name:str=''):
        self.neatmap_pipeline.registration_state.setText('Progressing ...')
        with open(template_file) as f:
            doc = json.load(f)
            template = sitk.ReadImage(os.path.join(ROOT_DIR, 'data', doc['file_name']))
            template_pixel_size = doc['voxel_size']
            atlas = None
            if 'atlas_file_name' in doc:
                atlas = sitk.ReadImage(os.path.join(ROOT_DIR, 'data', doc['atlas_file_name']))

        template.SetSpacing([1, 1, 1])
        input_file = os.path.join(output_path, 'Thumbnail_{}.json'.format(template_pixel_size))
        if os.path.exists(input_file):
            image = read_freesia2_image(input_file)
            image = sitk.Clamp((sitk.Log(sitk.Cast(image, sitk.sitkFloat32)) - 4.6) * 39.4, sitk.sitkFloat32, 0, 255)
            image.SetSpacing([1, 1, 1])
        else:
            brain_image_files = []
            with open(image_list_file) as f:
                doc_ = json.load(f)
                for i in doc_['images']:
                    brain_image_files.append(os.path.join(os.path.dirname(image_list_file), doc_['image_path'], i['file_name']))
                pixel_size = doc_['pixel_size']
                group_size = doc_['group_size']
            
            image = []
            scale = pixel_size / template_pixel_size
            for f in brain_image_files:
                im = sitk.ReadImage(f)
                im.SetSpacing([pixel_size for i in range(2)])
                size = [int(im.GetSize()[i] * scale) for i in range(2)]
                im = sitk.Resample(im, size, sitk.Transform(), sitk.sitkLinear, [0, 0], [template_pixel_size for i in range(2)])
                image.append(im)
            image = sitk.JoinSeries(image)
            size = [image.GetSize()[0], image.GetSize()[1], int(image.GetSize()[2] * scale)]
            image.SetSpacing([template_pixel_size, template_pixel_size, pixel_size])
            image = sitk.Resample(image, size, sitk.Transform(), sitk.sitkLinear, [0, 0, 0], [template_pixel_size for i in range(3)])
            image.SetSpacing([1, 1, 1])
            write_freesia2_image(image, output_path, 'Thumbnail_{}'.format(template_pixel_size), template_pixel_size,
                                int(group_size * scale))
            image = sitk.Clamp((sitk.Log(sitk.Cast(image, sitk.sitkFloat32)) - 4.6) * 39.4, sitk.sitkFloat32, 0, 255)
        out, tf, inv_tf = get_align_transform(image, template, [os.path.join(PARAMETER_DIR, self.params['registration_param'])],
                                            visormap_param=self.params, inverse_transform=True)
        sitk.WriteImage(out, os.path.join(output_path, 'registered.mha'))
        df = sitk.TransformToDisplacementField(tf, sitk.sitkVectorFloat32, template.GetSize())
        sitk.WriteImage(df, os.path.join(output_path, 'deformation_{}.mhd'.format(output_name)))
        df = sitk.TransformToDisplacementField(inv_tf, sitk.sitkVectorFloat32, image.GetSize())
        sitk.WriteImage(df, os.path.join(output_path, 'inverse_deformation_{}.mhd'.format(output_name)))
        if atlas is not None:
            atlas.SetSpacing([1, 1, 1])
            atlas = sitk.Resample(atlas, image, tf, sitk.sitkNearestNeighbor)
            atlas = sitk.Flip(atlas, [False, True, False])
            atlas_path = os.path.join(output_path, 'atlas')
            if not os.path.exists(atlas_path):
                os.mkdir(atlas_path)
            sitk.WriteImage(atlas, os.path.join(atlas_path, 'deformed_atlas_{}.mhd'.format(output_name)))
            if 'atlas' in doc:
                atlas_info = doc['atlas'][0]
                atlas_info['annotation_path'] = os.path.relpath(os.path.join(atlas_path, 'deformed_atlas_{}.raw'.format(output_name)),
                                                                os.path.dirname(input_file))
                atlas_info['image_dimension'] = '{} {} {}'.format(*atlas.GetSize())
                shutil.copy(os.path.join(ROOT_DIR, 'data', 'atlas_data', atlas_info['structures_path']), atlas_path)
                atlas_info['structures_path'] = os.path.relpath(os.path.join(atlas_path, atlas_info['structures_path']),
                                                                os.path.dirname(input_file))
                with open(input_file, 'r') as f:
                    d = json.load(f)
                d['freesia_project']['atlas'] = atlas_info
                with open(input_file, 'w') as f:
                    json.dump(d, f, indent=4)
                #atlas_info = {'atlas': [atlas_info]}
                #with open(os.path.join(atlas_path, 'freesia-atlas.json'), 'w') as f_:
                #    json.dump(atlas_info, f_, indent=4)
        self.signal.state.emit(self.neatmap_pipeline.registration_state)

    def segmentation_slice(self, data_path, post_path, slices, flip=False):
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
                self.signal.stop_pipeline.emit(self.neatmap_pipeline.stop)
                self.signal.progress_update.emit(self.neatmap_pipeline.BrainSlice_progressBar, int((i+1) / floor(len(brainimage_list)/slices) * 100))
                slice_list = brainimage_list[slices*i:slices*(i+1)]
                slice_image = np.zeros((slices,x,y)).astype('uint16')
                for j in range(len(slice_list)):
                    im = tifffile.imread(os.path.join(BrainImage_root, slice_list[j]))
                    if flip:
                        im = np.fliplr(im)
                    slice_image[j,: ,:] = im
                Save_path = os.path.join(save_root,str(i).zfill(5)+'.tif')
                tifffile.imwrite(Save_path,slice_image, compress=2)
        self.signal.state.emit(self.neatmap_pipeline.slice_state)

    def BrainImage2Spot(self, BrainImage_root,csv_root1):
        if not os.path.exists(csv_root1):
            os.mkdir(csv_root1)
        brainimage_list = os.listdir(BrainImage_root)
        brainimage_list.sort()
        save_file = open(os.path.join(csv_root1,'total.txt'),'w+')
        save_file.writelines('Id X Y Z Area\n')
        for i in range(0,len(brainimage_list)):
            self.signal.stop_pipeline.emit(self.neatmap_pipeline.stop)
            self.signal.progress_update.emit(self.neatmap_pipeline.CellCounts_progressBar, int((i+1) / len(brainimage_list) * 100))
            brainimage_path = os.path.join(BrainImage_root, brainimage_list[i])
            binary_image = tifffile.imread(brainimage_path)
            labeled_img = measure.label(binary_image, connectivity=1)
            properties = measure.regionprops(labeled_img)
            centroid_list = []
            area_list = []
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
                    content = str(j) + ' ' + str(x) + ' '+ str(y) +' ' + str(z_index) +' ' + str(area) + '\n'
                    save_file.writelines(content)
        save_file.close()
        self.signal.state.emit(self.neatmap_pipeline.counts_state)

    def Spot_csv(self, total_path, csv_root, brainimage2d_num, group_num):
        if not os.path.exists(csv_root):
            os.mkdir(csv_root)
        f = open(total_path,'r')
        spots = []
        for spot in f.readlines()[1:]:
            a = spot.strip().split(' ')
            a = np.array(a).astype(dtype=int).tolist()
            spots.append(a)
        # print(spots)

        # group with z = spots[3] , for every 6.25 slices
        for i in range(floor(brainimage2d_num/group_num)):
            self.signal.stop_pipeline.emit(self.neatmap_pipeline.stop)
            self.signal.progress_update.emit(self.neatmap_pipeline.ExportTable_progressBar, int((i+1) / floor(brainimage2d_num/group_num) * 100))
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
                        x ,y ,z = str(spots[j][1]*int(self.params['voxel_size'])), str(spots[j][2]*int(self.params['voxel_size'])), str(spots[j][3]*int(self.params['voxel_size']))
                        writeline = [x,y,z,'um','Spot','Position','1',str(count)]
                        count += 1
                        csv_write.writerow(writeline)
        self.signal.state.emit(self.neatmap_pipeline.export_state)
        self.neatmap_pipeline.start.setEnabled(True)
        self.neatmap_pipeline.stop.setEnabled(False)

    def Pipeline(self):

        self.work_queue = queue.Queue()
        self.work_function_queue = queue.Queue()

        ## Run Brain image
        json_file = os.path.join(self.neatmap_pipeline.DataRoot_lineEdit.text(), '..', 'freesia_4.0_'+ self.neatmap_pipeline.comboBox405.currentText() + '_405nm_10X.json')
        if os.path.exists(json_file):
            json_path = json_file
        else:
            json_path = os.path.join(self.neatmap_pipeline.DataRoot_lineEdit.text(), '..', 'freesia_4.0_'+ self.neatmap_pipeline.comboBox488.currentText() + '_488nm_10X.json')

        with open(json_path) as f:
            brain = json.load(f)
            images = brain['images']
            total_num = len(images)
            
        if self.neatmap_pipeline.buttonGroup.checkedButton().text() == '561nm':
            select_channel = self.neatmap_pipeline.comboBox561.currentText()
        elif self.neatmap_pipeline.buttonGroup.checkedButton().text() == '488nm':
            select_channel = self.neatmap_pipeline.comboBox488.currentText()
        elif self.neatmap_pipeline.buttonGroup.checkedButton().text() == '405nm':
            select_channel = self.neatmap_pipeline.comboBox405.currentText()

        self.neatmap_pipeline.start.setEnabled(False)
        self.neatmap_pipeline.stop.setEnabled(True)
        ## Run whole brain segmentation   
        channel = self.neatmap_pipeline.buttonGroup.checkedButton().text()
        patch_path = self.neatmap_pipeline.SaveRoot_lineEdit.text()
        snapshot = self.neatmap_pipeline.SnapshotPath_lineEdit.text()
        ## Run splice
        pred_root = os.path.join(self.neatmap_pipeline.SaveRoot_lineEdit.text(), 'whole_predications_' + channel)

        ## Run registration
        reconstruction_root = os.path.join(self.neatmap_pipeline.DataRoot_lineEdit.text(), '..', '..', '..', 'Reconstruction')
        channel405 = self.neatmap_pipeline.comboBox405.currentText()
        channel488 = self.neatmap_pipeline.comboBox488.currentText()
        image_list_json = os.path.join(reconstruction_root, 'BrainImage', 'freesia_4.0_'+ channel488 +'_488nm_10X.json')
        if os.path.exists(image_list_json):
            image_list_file = image_list_json
        else:
            image_list_file = os.path.join(reconstruction_root, 'BrainImage', 'freesia_4.0_'+ channel405 +'_405nm_10X.json')
        output_path = os.path.join(self.neatmap_pipeline.SaveRoot_lineEdit.text(), 'BrainRegistration')
        os.makedirs(output_path, exist_ok=True)
        template_file = 'pages/Registration/data/ccf_v3_template.json'
        output_name = self.params['registration_output_name']

        ## Run brain slice
        post_path = self.neatmap_pipeline.SaveRoot_lineEdit.text()
        data_path = os.path.join(post_path, 'whole_brain_pred_post_filter')

        ## Run cell counting
        BrainImage_root = os.path.join(self.neatmap_pipeline.SaveRoot_lineEdit.text(), 'whole_brain_pred_3d')
        csv_root = os.path.join(self.neatmap_pipeline.SaveRoot_lineEdit.text(), 'whole_brain_cell_counts')

        ## Run export csv file
        csv_root = os.path.join(self.neatmap_pipeline.SaveRoot_lineEdit.text(), 'whole_brain_cell_counts')
        save_csv_root = os.path.join(csv_root, 'Thumbnail_CSV')
        total_path = os.path.join(csv_root, 'total.txt')

        if self.neatmap_pipeline.buttonGroup.checkedButton().text() == '561nm':
            function_list = [self.brain2dto3d, self.cut_workthread, self.inference_whole_brain_seg, self.SpliceSegPatch,
                            self.post_workthread,  self.register_brain, self.segmentation_slice, self.BrainImage2Spot, self.Spot_csv]
            args_list = [(total_num, select_channel), (), (channel, patch_path, snapshot), (total_num, pred_root, channel), (), (image_list_file, output_path, template_file, output_name),
                        (data_path, post_path, int(self.params['Thickness'])), (BrainImage_root, csv_root), (total_path, save_csv_root, total_num, float(self.params['group_num']))]
        elif self.neatmap_pipeline.buttonGroup.checkedButton().text() == '488nm':
            function_list = [self.brain2dto3d, self.cut_workthread, self.inference_whole_brain_seg, self.SpliceSegPatch]
            args_list = [(total_num, select_channel), (), (channel, patch_path, snapshot), (total_num, pred_root, channel), ()]    
        
        strat_step_index = 0
        if self.neatmap_pipeline.checkBox_1.isChecked():
            strat_step_index = 0
        elif self.neatmap_pipeline.checkBox_2.isChecked():
            strat_step_index = 2
        elif self.neatmap_pipeline.checkBox_3.isChecked():
            strat_step_index = 3
        elif self.neatmap_pipeline.checkBox_4.isChecked():
            strat_step_index = 4
        elif self.neatmap_pipeline.checkBox_5.isChecked():
            strat_step_index = 5
        elif self.neatmap_pipeline.checkBox_6.isChecked():
            strat_step_index = 6

        for i in range(strat_step_index, len(function_list)):
            self.work_queue.put(args_list[i])
            self.work_function_queue.put(function_list[i])

        self.works = WorkerThread(self.work_queue, self.work_function_queue)
        self.works.start()
