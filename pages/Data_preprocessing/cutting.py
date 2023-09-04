# Synthetic 3d brain slices were cut into 70 patches.
from Environment_ui import *

def single_cutting(files, index, save_path, cut_size, cut_index_x, cut_index_y, patch_weight_num, patch_hegiht_num):

    image = sitk.ReadImage(files)
    image_ROI = image[cut_index_x : cut_size*patch_weight_num + cut_index_x, cut_index_y : cut_size*patch_hegiht_num + cut_index_y]
    image_array = sitk.GetArrayFromImage(image_ROI)
    
    num = 1
    start_x, start_y= 0, 0
    crop_width = cut_size
    crop_height = cut_size
    while start_x < image_array.shape[1]:
        end_x = min(start_x + crop_width, image_array.shape[1])
        end_y = start_y + crop_height
        cropped_image = image_array[:, start_x:end_x, start_y:end_y]
        cropped_sitk_image = sitk.GetImageFromArray(cropped_image)
        os.makedirs(os.path.join(save_path, 'patchimage{}'.format(index)), exist_ok=True)
        sitk.WriteImage(cropped_sitk_image, os.path.join(save_path, 'patchimage{}'.format(index), 'Z{:05d}_patch_{}.tif'.format(index, num)))
        num += 1
        start_y += crop_height
        if start_y >= image_array.shape[2]:
            start_y = 0
            start_x += crop_width
