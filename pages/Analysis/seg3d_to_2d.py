from Environment_ui import *

def seg3d_to_2d(data_path, save_path, index=0):

    os.makedirs(save_path, exist_ok=True)
    for i in range(1, len(os.listdir(data_path)) + 1):
        image = sitk.ReadImage(os.path.join(data_path, 'Z{:05d}_filter.tif'.format(i)))
        image_array = sitk.GetArrayFromImage(image)
        z = image_array.shape[0]
        for j in range(z):
            image_array_2d = image_array[j, :, :]
            image_2d = sitk.GetImageFromArray(image_array_2d)
            sitk.WriteImage(image_2d, os.path.join(save_path, 'Z{:05d}_seg.tif'.format(index)))
            index += 1
            print('finished {} 2d image'.format(j))
        print('Finished {} image'.format(i))

# if __name__ == "__main__":
#     path = Splicing['save_splicing_path']
#     data_path = os.path.join(path, 'whole_brain_pred_post_filter')
#     save_path = os.path.join(path, 'whole_brain_pred_2d')
#     seg3d_to_2d(data_path, save_path)