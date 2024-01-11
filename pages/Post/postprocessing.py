from Environment_ui import *
from scipy import ndimage as ndi
from skimage import measure

def label_coords(binary_image):
    labeled_img = measure.label(binary_image, connectivity=1)
    properties = measure.regionprops(labeled_img)
    coords_list = []
    for pro in properties:
        coord = pro.coords
        coords_list.append(coord)

    return coords_list

def post_488nm(pred_result, seg_488nm):

    seg_488nm[seg_488nm == 255] = 300
    result = pred_result + seg_488nm
    result[result == 555] = 0
    result[result == 300] = 0

    # coords_lists = label_coords(result)
    # for i in range(len(coords_lists)):
    #     region_pixels = autofluo_image[[coord[0] for coord in coords_lists[i]], [coord[1] for coord in coords_lists[i]], [coord[2] for coord in coords_lists[i]]]
    #     mean_intensity = np.mean(region_pixels)
    #     if mean_intensity > 300:
    #         result[[coord[0] for coord in coords_lists[i]], [coord[1] for coord in coords_lists[i]], [coord[2] for coord in coords_lists[i]]] = 0
    return result

def remove_piont(image, min_size, connectivity=1, in_place=False):
    if in_place:
        out = image
    else:
        out = image.copy()

    if min_size == 0:  
        return out

    if out.dtype == bool:
        selem = ndi.generate_binary_structure(image.ndim, connectivity)
        ccs = np.zeros_like(image, dtype=np.int32)
        ndi.label(image, selem, output=ccs)
    else:
        ccs = out
    component_sizes = np.bincount(ccs.ravel())

    too_small = component_sizes <= min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out

def big_object_filter(filter_matrix, seg, limit):

    filter_big_seg = seg.copy()
    bool_matrix = filter_matrix >= limit
    index_matrix = np.where(bool_matrix == True)
    for ind in range(len(index_matrix[0])):
        filter_big_seg[index_matrix[0][ind], index_matrix[1][ind], index_matrix[2][ind]] = 0

    return filter_big_seg
