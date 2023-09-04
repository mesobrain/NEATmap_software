from Environment_ui import *

def split(image, index, save_root, param):
    for ind in range(int(param['Network_input_z'])):
        for i in range(len(image)):
            split_image = image[i][ind, :, :]
            split_image = sitk.GetImageFromArray(split_image)
            save_path = os.path.join(save_root, 'whole_brain_split/' + 'split_image{}'.format(index) + '/{}'.format(ind + 1))
            os.makedirs(save_path, exist_ok=True)
            sitk.WriteImage(split_image, os.path.join(save_path, 'image_{}.tif'.format(i + 1)))

# if __name__ == "__main__":
#     root = Splicing['whole_predications_path']
#     for ind in range(1, len(os.listdir(root)) + 1):
#         path = root + '/brain_predications_{}_swin_epoch10/VISoR256/'.format(ind) + Splicing['checkpoint_name']
#         num = len(os.listdir(path))
#         imagelist = []
#         for i in range(1, num + 1):
#             image = sitk.ReadImage(os.path.join(path, 'Z{:05d}'.format(ind) + '_patch_{}_pred.tif'.format(i)))
#             array = sitk.GetArrayFromImage(image)
#             imagelist.append(array)
#         split(imagelist, ind, save_root=Splicing['save_root'])
#         print('finished {} split'.format(ind))