from Environment_ui import *
from PySide2.QtWidgets import QWidget, QFileDialog, QTextBrowser, QPushButton, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import Signal, QObject
from pages.Whole_brain_seg.utils import test_each_brain
from pages.Whole_brain_seg.dataset_whole_brain import Whole_brain_dataset
from pages.Whole_brain_seg.swin_transform import swin_tiny_patch4_window8_256
from torch.utils.data import DataLoader
from tqdm import tqdm
from threading import Thread

class SignalStore(QObject):

    progress_update = Signal(int)
    patch_print = Signal(QTextBrowser, str)
    stop_seg = Signal(QPushButton)

class WholeBrainSeg(QWidget):
    def __init__(self, params):
        super(WholeBrainSeg, self).__init__()
        self.signal = SignalStore()
        self.signal.patch_print.connect(self.PrintText)
        self.signal.progress_update.connect(self.ProgressUpdate)
        self.signal.stop_seg.connect(self.check_stop)

        self.whole_brain_seg = QUiLoader().load('pages/Whole_brain_seg/whole_brain_segmentation.ui')
        self.whole_brain_seg.BrainPatchPath.setPlaceholderText("Select brain patch path")
        self.whole_brain_seg.SnapshotPath.setPlaceholderText('Select the weight parameters (pth file) after training.')
        self.whole_brain_seg.Stop.setEnabled(False)

        self.whole_brain_seg.Load.clicked.connect(self.LoadPatch)
        self.whole_brain_seg.Selet.clicked.connect(self.SeletSnapshot)

        self.whole_brain_seg.Start.clicked.connect(self.run_whole_brain_seg)
        self.whole_brain_seg.Stop.clicked.connect(self.stop_segment)

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
        self.params = params

        self.args = parser.parse_args(args=[])        

    def PrintText(self, fb, text):
        fb.appendPlainText(text)
    
    def ProgressUpdate(self, value):
        self.whole_brain_seg.progressBar.setValue(value)

    def stop_segment(self):
        self.whole_brain_seg.Stop.setText('Stopping')

    def check_stop(self, sp):
        if sp.text() == 'Stopping':
            os._exit(0)

    def LoadPatch(self):
        filepath = QFileDialog.getExistingDirectory(self.whole_brain_seg, 'Selet patch path')
        self.whole_brain_seg.BrainPatchPath.setText(filepath)
    
    def SeletSnapshot(self):
        filepath = QFileDialog.getExistingDirectory(self.whole_brain_seg, 'Selet snapshot path')
        self.whole_brain_seg.SnapshotPath.setText(filepath)

    def infer_each_brain(self, args, model, ind, test_save_path=None):
        test_data = args.Dataset(base_dir=args.volume_path, split=args.name_list, list_dir=args.list_dir)
        testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model.eval()
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            image, case_name = sampled_batch["image"], sampled_batch['case_name'][0]
            test_each_brain(image, model, patch_size=[args.img_size, args.img_size], test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)

        return self.signal.patch_print.emit(self.whole_brain_seg.PrintText, "Testing Finished {}".format(ind))

    def inference_whole_brain_seg(self, channel, patch_path, snapshot_path):
        test_path = os.path.join(patch_path, 'PatchImage_' + channel)
        for ind in range(1, len(os.listdir(test_path)) + 1):
            self.signal.stop_seg.emit(self.whole_brain_seg.Stop)
            self.signal.progress_update.emit(int(ind / len(os.listdir(test_path)) * 100))
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
        self.whole_brain_seg.Stop.setText('Finished')   
        self.whole_brain_seg.Start.setEnabled(True)

    def run_whole_brain_seg(self):

        if len(self.whole_brain_seg.BrainPatchPath.text()) == 0:
            QMessageBox.critical(
                self.whole_brain_seg,
                "Error",
                "Please select brain patch path."
            )
        elif len(self.whole_brain_seg.SnapshotPath.text()) == 0:
            QMessageBox.critical(
                self.whole_brain_seg,
                "Error",
                "Please select data save path."
            )
        elif self.whole_brain_seg.Channel561.isChecked() == False and self.whole_brain_seg.Channel488.isChecked() == False:
            QMessageBox.critical(
                self.whole_brain_seg,
                "Error",
                "Please select channel."
            )
            
        select_channel = self.whole_brain_seg.ChannelGroup.checkedButton().text()
        patch_path = self.whole_brain_seg.BrainPatchPath.text()
        snapshot = self.whole_brain_seg.SnapshotPath.text()

        thread = Thread(target=self.inference_whole_brain_seg, args=(select_channel, patch_path, snapshot))
        self.whole_brain_seg.Start.setEnabled(False)
        self.whole_brain_seg.Stop.setEnabled(True)
        thread.start()



