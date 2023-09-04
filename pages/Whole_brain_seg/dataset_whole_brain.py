from Environment_ui import *
from torch.utils.data import Dataset

class Transform(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image.astype(np.float32))
        sample = {'image': image}

class Whole_brain_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform 
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        vol_name = self.sample_list[idx].strip('\n')
        filepath = self.data_dir + "/{}.tif".format(vol_name)
        # data = h5py.File(filepath)
        image = tifffile.imread(filepath)
        image = image.astype(np.float32)
        sample = {'image': image}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample