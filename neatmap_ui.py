from Environment_ui import *
from PySide2.QtWidgets import QMainWindow, QApplication, QSplashScreen
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QPixmap, QColor, QFont, QDesktopServices, QIcon
from PySide2.QtCore import Qt, QElapsedTimer, QUrl
from pages.Settings.settings import Settings
from pages.Settings.default_param import default_param
from pages.Data_preprocessing.data_preprocessing import Datapreprocess
from pages.Whole_brain_seg.whole_brain_segmentation import WholeBrainSeg
from pages.Splice.splice import Splice
from pages.Post.post import PostProcess
from pages.Registration.registration import BrainRegistration
from pages.Analysis.analysis import CellCount

ROOT_DIR = os.path.dirname(__file__)
VERSION = '1.1'

class NEATmap(QMainWindow):
    def __init__(self) -> None:
        super(NEATmap, self).__init__()
        self.neatmap_ui = QUiLoader().load('neatmap.ui')
        self.neatmap_ui.actionHelp.triggered.connect(self.show_user_guide)
        self.neatmap_ui.Settings.clicked.connect(self.Settings)
        self.neatmap_ui.datapro.clicked.connect(self.Datapreprocessing)
        self.neatmap_ui.WholeBrainSeg.clicked.connect(self.WholeBrainSeg)
        self.neatmap_ui.Splice.clicked.connect(self.Splice)
        self.neatmap_ui.Post.clicked.connect(self.Postprocessing)
        self.neatmap_ui.Registration.clicked.connect(self.Registration)
        self.neatmap_ui.Analysis.clicked.connect(self.Analysis)
        self.neatmap_ui.setWindowTitle('NEATmap {}'.format(VERSION))

        self.neatmap_params = default_param

    def show_user_guide(self):
        try:
            desktop_services = QDesktopServices()
            desktop_services.openUrl(QUrl().fromLocalFile(os.path.join(ROOT_DIR, 'doc', 'User_guide.pdf')))
        except:
            print('Failed to open user guide.')

    def Settings(self):
        self.set = Settings()
        self.set.settings.show()
        self.set.settings.Save.clicked.connect(self.get_params)
        self.set.settings.Back.clicked.connect(self.SettingsBackHomepage)
        self.neatmap_ui.hide()

    def get_params(self):
        self.set = Settings()
        self.neatmap_params = self.set.params
        self.neatmap_ui.show()

    def SettingsBackHomepage(self):
        self.set.settings.hide()
        self.neatmap_ui.show()

    def Datapreprocessing(self):
        self.datapreprocessing = Datapreprocess(params=self.neatmap_params)
        self.datapreprocessing.data_preprocess.show()
        self.datapreprocessing.data_preprocess.back.clicked.connect(self.DataproBackHomepage)
        self.neatmap_ui.hide()

    def DataproBackHomepage(self):
        self.datapreprocessing.data_preprocess.hide()
        self.neatmap_ui.show()

    def WholeBrainSeg(self):
        self.wholebrainseg = WholeBrainSeg(params=self.neatmap_params)
        self.wholebrainseg.whole_brain_seg.show()
        self.wholebrainseg.whole_brain_seg.Back.clicked.connect(self.WholeBrainSegBackHomepage)
        self.neatmap_ui.hide()

    def WholeBrainSegBackHomepage(self):
        self.wholebrainseg.whole_brain_seg.hide()
        self.neatmap_ui.show()
    
    def Splice(self):
        self.splicepacth = Splice(params=self.neatmap_params)
        self.splicepacth.splice.show()
        self.splicepacth.splice.Back.clicked.connect(self.SpliceBackHomepage)
        self.neatmap_ui.hide()

    def SpliceBackHomepage(self):
        self.splicepacth.splice.hide()
        self.neatmap_ui.show()

    def Postprocessing(self):
        self.postprocessing = PostProcess(params=self.neatmap_params)
        self.postprocessing.post_process.show()
        self.postprocessing.post_process.Back.clicked.connect(self.PostBackHomepage)
        self.neatmap_ui.hide()

    def PostBackHomepage(self):
        self.postprocessing.post_process.hide()
        self.neatmap_ui.show()

    def Registration(self):
        self.brainregistration = BrainRegistration(params=self.neatmap_params)
        self.brainregistration.registration.show()
        self.brainregistration.registration.Back.clicked.connect(self.RegistrationBackHomepage)
        self.neatmap_ui.hide()
    
    def RegistrationBackHomepage(self):
        self.brainregistration.registration.hide()
        self.neatmap_ui.show()

    def Analysis(self):
        self.analysis = CellCount(params=self.neatmap_params)
        self.analysis.cellcount.show()
        self.analysis.cellcount.Back.clicked.connect(self.AnalysisBackHomepage)
        self.neatmap_ui.hide()

    def AnalysisBackHomepage(self):
        self.analysis.cellcount.hide()
        self.neatmap_ui.show()
        
if __name__ == "__main__":
    envpath = r'C:\Users\Weijie\.conda\envs\sitkpy\Lib\site-packages\PySide2\plugins\platforms'
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = envpath
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    app.setWindowIcon(QIcon('figure/logo.png'))
    splash_image = QPixmap(os.path.join(ROOT_DIR, 'figure/splash_neatmap.png'))
    splash = QSplashScreen(splash_image)
    splash.setFont(QFont("Arial", 12))
    splash.showMessage('NEATmap {}'.format(VERSION), color=QColor(255, 255, 255),
                       alignment=Qt.AlignRight | Qt.AlignBottom)
                    
    splash.show()
    delayTime = 5
    timer = QElapsedTimer()
    timer.start()
    while (timer.elapsed() < (delayTime * 1000)):
        app.processEvents()
    neatmap = NEATmap()
    neatmap.neatmap_ui.show()
    splash.finish(neatmap)
    sys.exit(app.exec_())       