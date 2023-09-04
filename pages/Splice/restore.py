from Environment_ui import *

def concat(image_list, save_path, index, param):
    start_x, start_y= 0, 0
    restored_array = np.zeros((int(param['Network_input_z']), int(param['Patch_weight_num'])*int(param['Network_input_x_y']), 
                                int(param['Patch_height_num'])*int(param['Network_input_x_y'])), dtype=np.uint16)
    for i in range(len(image_list)):
        crop_shape = image_list[i][1].shape
        restored_array[:, start_x:start_x+crop_shape[1], start_y:start_y+crop_shape[2]] = image_list[i][1].transpose([0, 2, 1])

        start_x += crop_shape[1]
        if start_x >= int(param['Patch_weight_num'])*int(param['Network_input_x_y']):
            start_x = 0
            start_y += crop_shape[1]

    depth, row, col = restored_array.shape
    x_zeros_up = np.zeros((depth, row, int(param['Cut_index_y'])))
    x_zeros_down = np.zeros((depth, row, int(param['Brain_height']) - (restored_array.shape[2] + int(param['Cut_index_y']))))
    y_zeros = np.zeros((depth, (int(param['Brain_weight']) - row) // 2, int(param['Brain_height'])))
    img_resize = np.concatenate((restored_array, x_zeros_down), axis = 2)
    img_resize = np.concatenate((x_zeros_up, img_resize), axis = 2)
    img_resize = np.concatenate((img_resize, y_zeros), axis = 1)
    img_resize = np.concatenate((y_zeros, img_resize), axis = 1)
    img_resize = img_resize.transpose([0, 2, 1])
    restored_image = sitk.GetImageFromArray(img_resize)
    restored_image = sitk.Cast(restored_image, sitk.sitkUInt16)

    sitk.WriteImage(restored_image, os.path.join(save_path, 'Z{:05d}_seg.tif'.format(index)))

    return save_path

def load(path, index, total_patch_num):
    images = []
    for i in range(1, total_patch_num + 1):
        image_path = os.path.join(path, 'Z{:05d}_patch_{}_pred.tif'.format(index, i))   
        image = sitk.ReadImage(image_path)
        array = sitk.GetArrayFromImage(image)
        images.append((image_path, array)) 
    sorted_images = sorted(images, key=lambda x: int(re.search(r'\d+', x[0]).group()))
    return sorted_images
    

def create_residual_image(save_path, total_num, resuidual_z, param):

    resuidual_array = np.zeros((resuidual_z, int(param['Brain_height']), int(param['Brain_weight'])))
    resuidual_image = sitk.GetImageFromArray(resuidual_array)
    resuidual_image = sitk.Cast(resuidual_image, sitk.sitkUInt16)
    sitk.WriteImage(resuidual_image, save_path + '/Z{:05d}_seg.tif'.format(total_num))
   