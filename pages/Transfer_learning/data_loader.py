from Environment_ui import *
from torch.utils.data import Dataset

class VISoR_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split):
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "valid":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            label[label==255] = 1
        sample = {'image': image, 'label': label}

        image, label = sample['image'], sample['label']
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32)).squeeze(0)
        sample = {'image': image, 'label': label.long()}
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample