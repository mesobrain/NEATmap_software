from Environment_ui import *
from PySide2.QtWidgets import QWidget, QFileDialog, QPushButton, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import Signal, QObject
from threading import Thread
from pages.Registration.brain_registration import read_freesia2_image, write_freesia2_image
from pages.Registration.elastix_files import get_align_transform

ROOT_DIR = os.path.dirname(__file__)
PARAMETER_DIR = 'pages/Registration/parameters'

class SignalStore(QObject):
    stop_registration = Signal(QPushButton)

class BrainRegistration(QWidget):
    def __init__(self, params) -> None:
        super(BrainRegistration, self).__init__()
        self.signal = SignalStore()
        self.signal.stop_registration.connect(self.CheckStop)

        self.registration = QUiLoader().load('pages/Registration/Registration.ui')
        self.registration.Stop.setEnabled(False)
        self.registration.DataRootLine.setPlaceholderText("Select BrainImgae/4.0 for the reconstruction data path")
        self.registration.SaveRootLine.setPlaceholderText("Select the path to save the data")

        self.registration.LoadData.clicked.connect(self.LoadData)
        self.registration.SeletSave.clicked.connect(self.SeletSave)
        self.registration.Start.clicked.connect(self.run_registration)
        self.registration.Stop.clicked.connect(self.Stop)

        self.registration.comboBox561.addItems(['C1', 'C2', 'C3', 'C4'])
        self.registration.comboBox488.addItems(['C1', 'C2', 'C3', 'C4'])

        self.params = params

    def LoadData(self):
        filepath = QFileDialog.getExistingDirectory(self.registration, 'Selet data root')
        self.registration.DataRootLine.setText(filepath)

    def SeletSave(self):
        filepath = QFileDialog.getExistingDirectory(self.registration, 'Selet save root')
        self.registration.SaveRootLine.setText(filepath)

    def Stop(self):
        self.registration.Stop.setText('Stopping')

    def CheckStop(self, sp):
        if sp.text() == 'Stopping':
            os._exit(0)

    def run_registration(self):

        if len(self.registration.DataRootLine.text()) == 0:
            QMessageBox.critical(
                self.registration,
                "Error",
                "Please select raw data path."
            )
        elif len(self.registration.SaveRootLine.text()) == 0:
            QMessageBox.critical(
                self.registration,
                "Error",
                "Please select save path."
            )

        reconstruction_root = os.path.join(self.registration.DataRootLine.text(), 'Reconstruction')
        channel488 = self.registration.comboBox488.currentText()
        image_list_file = os.path.join(reconstruction_root, 'BrainImage', 'freesia_4.0_'+ channel488 +'_488nm_10X.json')
        output_path = os.path.join(self.registration.SaveRootLine.text(), 'BrainRegistration')
        os.makedirs(output_path, exist_ok=True)
        template_file = 'pages/Registration/data/ccf_v3_template.json'
        output_name = self.params['registration_output_name']
        
        thread = Thread(target=self.register_brain, 
                        args=(image_list_file, output_path, template_file, output_name))
        thread.start()

    def register_brain(self, image_list_file:str, output_path: str, template_file: str, output_name:str=''):
        self.registration.Start.setEnabled(False)
        self.registration.Stop.setEnabled(True)
        self.signal.stop_registration.emit(self.registration.Stop)
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
        self.registration.Stop.setText('Finished')
        self.registration.Stop.setEnabled(True)
        self.registration.Start.setEnabled(True)
