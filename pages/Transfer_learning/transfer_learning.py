from Environment_ui import *
from PySide2.QtWidgets import QWidget, QFileDialog, QPushButton, QLabel, QTextBrowser
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import Signal, QObject
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from pages.Data_preprocessing.cutting import single_cutting
from pages.Transfer_learning.loss import DiceLoss
from pages.Transfer_learning.data_loader import VISoR_dataset
from pages.Transfer_learning.utils import ConfusionMatrix, weights_init
from pages.Whole_brain_seg.swin_transform import swin_tiny_patch4_window8_256

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
    print = Signal(QTextBrowser, str)
    stop_transferlearning = Signal(QPushButton)

class TransferLearning(QWidget):
    def __init__(self, params):
        super(TransferLearning, self).__init__()
        self.signal = SignalStore()
        self.signal.print.connect(self.printToGui)

        self.transfer_learning = QUiLoader().load('pages/Transfer_learning/transfer_learning.ui')
        self.transfer_learning.DataPath.setPlaceholderText('Select the training image data path.')
        self.transfer_learning.LabelPath.setPlaceholderText('Select the training label data path.')
        self.transfer_learning.SnapshotPath.setPlaceholderText('Select the weight parameters (pth file) after training.')
        self.transfer_learning.DataLoad.clicked.connect(self.SeletData)
        self.transfer_learning.LabelLoad.clicked.connect(self.SeletLabel)
        self.transfer_learning.SnapshotSelect.clicked.connect(self.SeletSnapshot)
        self.transfer_learning.Start.clicked.connect(self.Transfer)

        self.transfer_learning.CudacomboBox.addItems(['0', '1', '2', '3','False'])
        self.transfer_learning.Stop.setEnabled(False)

        self.params = params

    def printToGui(self, fb, text):
        fb.appendPlainText(text)

    def SeletData(self):
        filepath = QFileDialog.getExistingDirectory(self.transfer_learning, 'Selet data path')
        self.transfer_learning.DataPath.setText(filepath)

    def SeletLabel(self):
        filepath = QFileDialog.getExistingDirectory(self.transfer_learning, 'Selet label path')
        self.transfer_learning.LabelPath.setText(filepath)

    def SeletSnapshot(self):
        filepath = QFileDialog.getExistingDirectory(self.transfer_learning, 'Selet snapshot path')
        self.transfer_learning.SnapshotPath.setText(filepath)

    def Cutting(self, data_path, label_path, cut_size, cut_index_x, cut_index_y, patch_weight_num, 
            patch_hegiht_num, index, train_path=None, train_label_path=None, cut_label=False):

        if train_path is not None:
            save_path = os.path.join(train_path, 'image_patch')
            os.makedirs(save_path, exist_ok=True)
        if train_label_path is not None:
            save_path = os.path.join(train_label_path, 'label_patch')
            os.makedirs(save_path, exist_ok=True)

        if cut_label:
            name = 'Z{:05d}_seg'.format(index)
            image = os.path.join(label_path, name + '.tif')
        else:
            name = 'Z{:05d}'.format(index)
            image = os.path.join(data_path, name + '.tif')
        single_cutting(image, index, save_path, cut_size, cut_index_x, cut_index_y, patch_weight_num, patch_hegiht_num)
        self.signal.print.emit(self.transfer_learning.PrintTransferLearning, 'Finished {} cut'.format(index))

    def run_image_cutting(self, index):

        self.transfer_learning.Start.setEnabled(False)
        self.transfer_learning.Stop.setEnabled(True)
        data_path = self.transfer_learning.DataPath.text()
        label_path = self.transfer_learning.LabelPath.text()
        cut_size = int(self.params['Network_input_x_y'])
        cut_index_x = int(self.params['Cut_index_x'])
        cut_index_y = int(self.params['Cut_index_y'])
        patch_weight_num = int(self.params['Patch_weight_num'])
        patch_hegiht_num = int(self.params['Patch_height_num'])
        train_path = os.path.join(data_path, '..')
        self.Cutting(data_path, label_path, cut_size, cut_index_x, cut_index_y, 
                    patch_weight_num, patch_hegiht_num, index, train_path=train_path)

    def image_cut_workthread(self):

        root = self.transfer_learning.DataPath.text()
        data_list = [k for k in range(1, len(os.listdir(root)) + 1)]
        with ThreadPoolExecutor(max_workers=10) as pool:
            pool.map(self.run_image_cutting, data_list)

    def run_label_cutting(self, index):

        data_path = self.transfer_learning.DataPath.text()
        label_path = self.transfer_learning.LabelPath.text()
        cut_size = int(self.params['Network_input_x_y'])
        cut_index_x = int(self.params['Cut_index_x'])
        cut_index_y = int(self.params['Cut_index_y'])
        patch_weight_num = int(self.params['Patch_weight_num'])
        patch_hegiht_num = int(self.params['Patch_height_num'])
        train_label_path = os.path.join(label_path, '..')
        self.Cutting(data_path, label_path, cut_size, cut_index_x, cut_index_y, 
                    patch_weight_num, patch_hegiht_num, index, train_label_path=train_label_path, cut_label=True)

    def label_cut_workthread(self):

        root = self.transfer_learning.DataPath.text()
        data_list = [k for k in range(1, len(os.listdir(root)) + 1)]
        with ThreadPoolExecutor(max_workers=10) as pool:
            pool.map(self.run_label_cutting, data_list)

    def sort(self, image_path, seg_path, save_image_path, save_label_path):

        os.makedirs(save_image_path, exist_ok=True)
        os.makedirs(save_label_path, exist_ok=True)
        ind = 1
        for i in range(1, len(os.listdir(image_path)) + 1):
            for j in range(1, int(self.params['Patch_weight_num']) * int(self.params['Patch_height_num']) + 1):
                image = tifffile.imread(os.path.join(image_path, 'patchimage{}'.format(i), 'Z{:05d}_patch_{}.tif'.format(i, j)))
                seg = tifffile.imread(os.path.join(seg_path, 'patchimage{}'.format(i), 'Z{:05d}_patch_{}.tif'.format(i, j)))
                tifffile.imwrite(os.path.join(save_image_path, 'image_{}.tif'.format(ind)), image.astype('uint16'))
                tifffile.imwrite(os.path.join(save_label_path, 'label_{}.tif'.format(ind)), seg.astype('uint16'))
                self.signal.print.emit(self.transfer_learning.PrintTransferLearning, 'Finished {} file'.format(ind))
                ind += 1

    def load_tif2array(self, file):
        
        image = sitk.ReadImage(file)
        image = sitk.Cast(image, sitk.sitkFloat32)
        image_array = sitk.GetArrayFromImage(image)
        image_array = np.transpose(image_array, [0, 2, 1])
        return image_array

    def get_train_data(self, image_path, label_path, save_path):

        os.makedirs(save_path, exist_ok=True)
        for i in range(1, len(os.listdir(image_path)) + 1):
            image_file = os.path.join(image_path, 'image_{}.tif').format(i)
            label_file = os.path.join(label_path, 'label_{}.tif').format(i)
            if os.path.isfile(image_file) and os.path.isfile(label_file):
                image_array = self.load_tif2array(image_file)
                label_array = self.load_tif2array(label_file)
                label_array[label_array == 255] = 1
                np.savez(os.path.join(save_path, 'data_patch_{}.npz').format(i), image=image_array, label=label_array)
                self.signal.print.emit(self.transfer_learning.PrintTransferLearning, 'Finished {} data'.format(i))
            else:
                continue

    def write_text(self, train_data_path, save_text_path):

        os.makedirs(save_text_path, exist_ok=True)
        name_list = os.listdir(train_data_path)
        name = []
        for vlaue in name_list:
            protions = vlaue.split('.')
            line = '{}\n'.format(protions[0])
            name.append(line)
        total_num = len(name)
        train_num = int(len(name)*0.9)
        train_index = name[0:train_num]
        valid_index = name[train_num:total_num]
        with open(os.path.join(save_text_path, 'train.txt'), 'w') as f:
            f.writelines(train_index)
        with open(os.path.join(save_text_path, 'valid.txt'), 'w') as v:
            v.writelines(valid_index)
        self.signal.print.emit(self.transfer_learning.PrintTransferLearning, 'Finished write text')

    def train(self, net, checkpoint, pre_train, device):
        warnings.filterwarnings("ignore")
        if pre_train:
            weight = torch.load(checkpoint, map_location='cpu')
            net.load_state_dict(weight, strict=False)
            self.signal.print.emit(self.transfer_learning.PrintTransferLearning, 'net pre-train !')
        net.train()
        save_checkpoint = os.path.join(os.path.join(self.transfer_learning.DataPath.text(), '..', 'transfer_checkpoint'))
        os.makedirs(save_checkpoint, exist_ok=True)
        logging.basicConfig(filename= save_checkpoint + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        img_size = int(self.params['Network_input_x_y'])
        base_lr = float(self.params['lr'])
        max_epochs = int(self.params['max_epochs'])
        num_classes = int(self.params['num_classes'])

        data_path = os.path.join(self.transfer_learning.DataPath.text(), '..', 'train_data')
        list_dir = os.path.join(self.transfer_learning.DataPath.text(), '..', 'lists')

        train_data = VISoR_dataset(base_dir=data_path, list_dir=list_dir, split="train")
        val_data = VISoR_dataset(base_dir=data_path, list_dir=list_dir, split="valid")

        train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1)

        self.signal.print.emit(self.transfer_learning.PrintTransferLearning, "The length of train set is: {}".format(len(train_loader)))
        self.signal.print.emit(self.transfer_learning.PrintTransferLearning, "The length of valid set is: {}".format(len(valid_loader)))

        optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

        criterion = nn.CrossEntropyLoss()
        dice_loss = DiceLoss(num_classes)
 
        iter_num = 0
        validate_every_n_epoch = 2
        max_epoch = max_epochs
        max_iterations = max_epochs * len(train_loader)   
        logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
        iterator = tqdm(range(max_epoch), ncols=70)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(train_loader):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                image_batch, label_batch = Variable(image_batch), Variable(label_batch)
                pred = net(image_batch)
                loss_ce = criterion(pred, label_batch[:].long())
                loss_dice = dice_loss(pred, label_batch, softmax=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_ = optimizer.param_groups[0]["lr"]
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1
                logging.info('iteration %d : loss : %f, loss_ce : %f' % (iter_num, loss.item(), loss_ce.item()))
                self.signal.print.emit(self.transfer_learning.PrintTransferLearning, 'iteration %d : loss : %f, loss_ce : %f' % (iter_num, loss.item(), loss_ce.item()))

            if epoch_num % validate_every_n_epoch ==0:
                confmat = ConfusionMatrix(num_classes)
                val_loader = tqdm(valid_loader, desc="Validate")
                val_iter_num = 0
                net.eval()
                for i, val_data in enumerate(val_loader):
                    val_image, val_label = val_data['image'], val_data['label']
                    val_image, val_label = val_image.to(device), val_label.to(device)
                    with torch.no_grad():
                        val_out = net(val_image)
                        val_out = torch.softmax(val_out, dim=1)
                        confmat.update(val_label.squeeze().flatten(), torch.argmax(val_out, dim=1).flatten())
                    val_iter_num = val_iter_num + 1
                    logging.info(confmat)

            save_interval = 5  # int(max_epoch/6)

            if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
                save_mode_path = os.path.join(save_checkpoint, 'epoch_' + str(epoch_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                self.signal.print.emit(self.transfer_learning.PrintTransferLearning, "save model to {}".format(save_mode_path))

            if epoch_num >= max_epoch - 1:
                save_mode_path = os.path.join(save_checkpoint, 'epoch_' + str(epoch_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                self.signal.print.emit(self.transfer_learning.PrintTransferLearning, "save model to {}".format(save_mode_path))
                iterator.close()
                break

        self.transfer_learning.Start.setEnabled(True)
        self.transfer_learning.Stop.setText('Finished')
        return self.signal.print.emit(self.transfer_learning.PrintTransferLearning, "Training Finished!") 

    def Transfer(self):

        self.work_queue = queue.Queue()
        self.work_function_queue = queue.Queue()

        ## Sort file
        image_path = os.path.join(self.transfer_learning.DataPath.text(), '..', 'image_patch')
        label_path = os.path.join(self.transfer_learning.LabelPath.text(), '..', 'label_patch')
        save_image_path = os.path.join(self.transfer_learning.DataPath.text(), '..', 'train_image')
        save_label_path = os.path.join(self.transfer_learning.DataPath.text(), '..', 'train_label')

        ## tif file transfer npz file
        save_data_path = os.path.join(self.transfer_learning.DataPath.text(), '..', 'train_data')

        ## Write text
        save_text_path = os.path.join(self.transfer_learning.DataPath.text(), '..', 'lists')

        ## Creat model
        pre_train = True
        checkpoint_path = os.path.join(self.transfer_learning.SnapshotPath.text(), 'epoch_' + str(int(self.params['max_epochs']) - 1) + '.pth')
        if self.transfer_learning.CudacomboBox.currentText() == 'False':
            device = torch.device('cpu')
        else:
            device = torch.device("cuda:{}".format(self.transfer_learning.CudacomboBox.currentText()) if torch.cuda.is_available() else 'cpu')
        net = swin_tiny_patch4_window8_256(num_classes=2).to(device)
        net = net.apply(weights_init)

        ## Run
        function_list = [self.image_cut_workthread, self.label_cut_workthread, self.sort, self.get_train_data, self.write_text, self.train]
        args_list = [(), (), (image_path, label_path, save_image_path, save_label_path), (save_image_path, save_label_path, save_data_path), 
                    (save_data_path, save_text_path), (net, checkpoint_path, pre_train, device)]

        for i in range(len(function_list)):
            self.work_queue.put(args_list[i])
            self.work_function_queue.put(function_list[i])

        self.works = WorkerThread(self.work_queue, self.work_function_queue)
        self.works.start()        