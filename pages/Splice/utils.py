from Environment_ui import *
from scipy.ndimage import gaussian_filter
from typing import List
from scipy.stats import norm
from scipy import ndimage as ndi
from itertools import combinations_with_replacement


def intensity_normalization(struct_img: np.ndarray, scaling_param: List):
    assert len(scaling_param) > 0

    if len(scaling_param) == 1:
        if scaling_param[0] < 1:
            print("intensity normalization: min-max normalization with NO absolute" + "intensity upper bound")
        else:
            print(f"intensity norm: min-max norm with upper bound {scaling_param[0]}")
            # struct_img[struct_img > scaling_param[0]] = struct_img.min()
            struct_img[struct_img > scaling_param[0]] = struct_img.min()
        strech_min = struct_img.min()
        strech_max = struct_img.max()
    elif len(scaling_param) == 2:
        m, s = norm.fit(struct_img.flat)
        strech_min = max(m - scaling_param[0] * s, struct_img.min())
        strech_max = min(m + scaling_param[1] * s, struct_img.max())
        struct_img[struct_img > strech_max] = strech_max
        struct_img[struct_img < strech_min] = strech_min
    elif len(scaling_param) == 4:
        img_valid = struct_img[np.logical_and(struct_img > scaling_param[2], struct_img < scaling_param[3])]
        assert (
            img_valid.size > 0
        ), f"Adjust intensity normalization parameters {scaling_param[2]} and {scaling_param[3]} to include the image with range {struct_img.min()}:{struct_img.max()}"  # noqa: E501
        m, s = norm.fit(img_valid.flat)
        strech_min = max(scaling_param[2] - scaling_param[0] * s, struct_img.min())
        strech_max = min(scaling_param[3] + scaling_param[1] * s, struct_img.max())
        struct_img[struct_img > strech_max] = strech_max
        struct_img[struct_img < strech_min] = strech_min
    assert (
        strech_min <= strech_max
    ), f"Please adjust intensity normalization parameters so that {strech_min}<={strech_max}"
    struct_img = (struct_img - strech_min + 1e-8) / (strech_max - strech_min + 1e-8)

    # print('intensity normalization completes')
    return struct_img

def image_smoothing_gaussian_slice_by_slice(struct_img, sigma, truncate_range=3.0):
    """
    wrapper for applying 2D Guassian smoothing slice by slice on a 3D image
    """
    structure_img_smooth = np.zeros_like(struct_img)
    for zz in range(struct_img.shape[0]):
        structure_img_smooth[zz, :, :] = gaussian_filter(
            struct_img[zz, :, :], sigma=sigma, mode="nearest", truncate=truncate_range
        )

    return structure_img_smooth

def single_fluorescent_view(im):
    assert len(im.shape) == 3

    im = im.astype(np.float32)
    im = (im - im.min()) / (im.max() - im.min())

    return im

def dot_3d_wrapper(struct_img: np.ndarray, s3_param: List):

    bw = np.zeros(struct_img.shape, dtype=bool)
    for fid in range(len(s3_param)):
        log_sigma = s3_param[fid][0]
        responce = -1 * (log_sigma ** 2) * ndi.filters.gaussian_laplace(struct_img, log_sigma)
        bw = np.logical_or(bw, responce > s3_param[fid][1])
    return bw

def segmentation_quick_view(seg: np.ndarray):
    seg = seg > 0
    seg = seg.astype(np.uint16)
    seg[seg > 0] = 255

    return seg

def dot_2d_slice_by_slice_wrapper(struct_img: np.ndarray, s2_param: List):
    """wrapper for 2D spot filter on 3D image slice by slice

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d numpy array, usually the image after smoothing
    s2_param: List
        [[scale_1, cutoff_1], [scale_2, cutoff_2], ....], e.g. [[1, 0.1]]
        or [[1, 0.12], [3,0.1]]: scale_x is set based on the estimated radius
        of your target dots. For example, if visually the diameter of the
        dots is usually 3~4 pixels, then you may want to set scale_x as 1
        or something near 1 (like 1.25). Multiple scales can be used, if
        you have dots of very different sizes. cutoff_x is a threshold
        applied on the actual filter reponse to get the binary result.
        Smaller cutoff_x may yielf more dots and fatter segmentation,
        while larger cutoff_x could be less permisive and yield less
        dots and slimmer segmentation.
    """
    bw = np.zeros(struct_img.shape, dtype=bool)
    for fid in range(len(s2_param)):
        log_sigma = s2_param[fid][0]
        responce = np.zeros_like(struct_img)
        for zz in range(struct_img.shape[0]):
            responce[zz, :, :] = -1 * (log_sigma ** 2) * ndi.filters.gaussian_laplace(struct_img[zz, :, :], log_sigma)
        bw = np.logical_or(bw, responce > s2_param[fid][1])
    return bw

def remove_edge(image_path, label_array, critical_value=112, interval=15):
	image = sitk.ReadImage(image_path)
	image_array = sitk.GetArrayFromImage(image)
	image_array[image_array > critical_value] = 255
	image_array[image_array <= critical_value] = 0
	binary_image = sitk.GetImageFromArray(image_array)
	binary_image = sitk.Cast(binary_image, sitk.sitkFloat32)
	edges = sitk.CannyEdgeDetection(binary_image, lowerThreshold=0.0, upperThreshold=40.0, variance = (5.0,5.0,5.0))
	edge_indexes = np.where(sitk.GetArrayViewFromImage(edges) == 1.0)
	physical_points = [edges.TransformIndexToPhysicalPoint([int(x), int(y), int(z)]) \
					for z,y,x in zip(edge_indexes[0], edge_indexes[1], edge_indexes[2])]
	# label_edge_indexes = np.where(label_array == 255)
	edge_array = np.zeros_like(image_array)
	for i in range(len(physical_points)):
		edge_z, edge_y, edge_x = int(physical_points[i][2]), int(physical_points[i][1]), int(physical_points[i][0])
		edge_array[edge_z, edge_y-interval:edge_y+interval, edge_x-interval:edge_x+interval] = 300

	remove_edge_array = label_array + edge_array
	remove_edge_array[remove_edge_array==555]=0
	remove_edge_array[remove_edge_array==300]=0
    
	print('Complete removal of boundary')

	return remove_edge_array

def filament_3d_wrapper(struct_img: np.ndarray, f3_param: List[List]):
    """wrapper for 3d filament filter

    Parameters:
    ------------
    struct_img: np.ndarray
        the image (should have been smoothed) to be segmented. The image has to be 3D.
    f3_param: List[List]
        [[scale_1, cutoff_1], [scale_2, cutoff_2], ....], e.g., [[1, 0.01]] or
        [[1,0.05], [0.5, 0.1]]. scale_x is set based on the estimated thickness of your
        target filaments. For example, if visually the thickness of the filaments is
        usually 3~4 pixels, then you may want to set scale_x as 1 or something near 1
        (like 1.25). Multiple scales can be used, if you have filaments of very
        different thickness. cutoff_x is a threshold applied on the actual filter
        reponse to get the binary result. Smaller cutoff_x may yielf more filaments,
        especially detecting more dim ones and thicker segmentation, while larger
        cutoff_x could be less permisive and yield less filaments and slimmer
        segmentation.

    Reference:
    ------------
    T. Jerman, et al. Enhancement of vascular structures in 3D and 2D angiographic
    images. IEEE transactions on medical imaging. 2016 Apr 4;35(9):2107-18.
    """
    assert len(struct_img.shape) == 3, "image has to be 3D"
    bw = np.zeros(struct_img.shape, dtype=bool)
    for fid in range(len(f3_param)):
        sigma = f3_param[fid][0]
        eigenvalues = absolute_3d_hessian_eigenvalues(struct_img, sigma=sigma, scale=True, whiteonblack=True)
        responce = compute_vesselness3D(eigenvalues[1], eigenvalues[2], tau=1)
        bw = np.logical_or(bw, responce > f3_param[fid][1])
    return bw


def filament_2d_wrapper(struct_img: np.ndarray, f2_param: List[List]):
    """wrapper for 2d filament filter

    Parameters:
    ------------
    struct_img: np.ndarray
        the image (should have been smoothed) to be segmented. The image is
        either 2D or 3D. If 3D, the filter is applied in a slice by slice
        fashion
    f2_param: List[List]
        [[scale_1, cutoff_1], [scale_2, cutoff_2], ....], e.g., [[1, 0.01]]
        or [[1,0.05], [0.5, 0.1]]. Here, scale_x is set based on the estimated
        thickness of your target filaments. For example, if visually the thickness
        of the filaments is usually 3~4 pixels, then you may want to set scale_x
        as 1 or something near 1 (like 1.25). Multiple scales can be used, if you
        have filaments of very different thickness. cutoff_x is a threshold applied
        on the actual filter reponse to get the binary result. Smaller cutoff_x may
        yielf more filaments, especially detecting more dim ones and thicker
        segmentation, while larger cutoff_x could be less permisive and yield less
        filaments and slimmer segmentation.

    Reference:
    ------------
    T. Jerman, et al. Enhancement of vascular structures in 3D and 2D angiographic
    images. IEEE transactions on medical imaging. 2016 Apr 4;35(9):2107-18.
    """
    bw = np.zeros(struct_img.shape, dtype=bool)

    if len(struct_img.shape) == 2:
        for fid in range(len(f2_param)):
            sigma = f2_param[fid][0]
            eigenvalues = absolute_3d_hessian_eigenvalues(struct_img, sigma=sigma, scale=True, whiteonblack=True)
            responce = compute_vesselness2D(eigenvalues[1], tau=1)
            bw = np.logical_or(bw, responce > f2_param[fid][1])
    elif len(struct_img.shape) == 3:
        mip = np.amax(struct_img, axis=0)
        for fid in range(len(f2_param)):
            sigma = f2_param[fid][0]

            res = np.zeros_like(struct_img)
            for zz in range(struct_img.shape[0]):
                tmp = np.concatenate((struct_img[zz, :, :], mip), axis=1)
                eigenvalues = absolute_3d_hessian_eigenvalues(tmp, sigma=sigma, scale=True, whiteonblack=True)
                responce = compute_vesselness2D(eigenvalues[1], tau=1)
                res[zz, :, : struct_img.shape[2] - 3] = responce[:, : struct_img.shape[2] - 3]
            bw = np.logical_or(bw, res > f2_param[fid][1])
    return bw

def compute_vesselness3D(eigen2, eigen3, tau):
    """backend for computing 3D filament filter"""

    lambda3m = copy.copy(eigen3)
    lambda3m[np.logical_and(eigen3 < 0, eigen3 > (tau * eigen3.min()))] = tau * eigen3.min()
    response = np.multiply(np.square(eigen2), np.abs(lambda3m - eigen2))
    response = divide_nonzero(27 * response, np.power(2 * np.abs(eigen2) + np.abs(lambda3m - eigen2), 3))

    response[np.less(eigen2, 0.5 * lambda3m)] = 1
    response[eigen2 >= 0] = 0
    response[eigen3 >= 0] = 0
    response[np.isinf(response)] = 0

    return response


def compute_vesselness2D(eigen2, tau):
    """backend for computing 2D filament filter"""

    Lambda3 = copy.copy(eigen2)
    Lambda3[np.logical_and(Lambda3 < 0, Lambda3 >= (tau * Lambda3.min()))] = tau * Lambda3.min()

    response = np.multiply(np.square(eigen2), np.abs(Lambda3 - eigen2))
    response = divide_nonzero(27 * response, np.power(2 * np.abs(eigen2) + np.abs(Lambda3 - eigen2), 3))

    response[np.less(eigen2, 0.5 * Lambda3)] = 1
    response[eigen2 >= 0] = 0
    response[np.isinf(response)] = 0

    return response

def absolute_eigenvaluesh(nd_array):
    """Computes the eigenvalues sorted by absolute value from the symmetrical matrix.

    Parameters:
    -------------
    nd_array: nd.ndarray
        array from which the eigenvalues will be calculated.

    Return:
    -------------
        A list with the eigenvalues sorted in absolute ascending order (e.g.
        [eigenvalue1, eigenvalue2, ...])
    """
    eigenvalues = np.linalg.eigvalsh(nd_array)
    sorted_eigenvalues = sortbyabs(eigenvalues, axis=-1)
    return [
        np.squeeze(eigenvalue, axis=-1)
        for eigenvalue in np.split(sorted_eigenvalues, sorted_eigenvalues.shape[-1], axis=-1)
    ]

def sortbyabs(a, axis=0):
    """Sort array along a given axis by the absolute value
    modified from: http://stackoverflow.com/a/11253931/4067734
    """
    index = list(np.ix_(*[np.arange(i) for i in a.shape]))
    index[axis] = np.abs(a).argsort(axis)
    return a[index]

def divide_nonzero(array1, array2):
    """
    Divides two arrays. Returns zero when dividing by zero.
    """
    denominator = np.copy(array2)
    denominator[denominator == 0] = 1e-10
    return np.divide(array1, denominator)

def absolute_3d_hessian_eigenvalues(
    nd_array: np.ndarray,
    sigma: float = 1,
    scale: bool = True,
    whiteonblack: bool = True,
):
    """
    Eigenvalues of the hessian matrix calculated from the input array sorted by
    absolute value.

    Parameters:
    ------------
    nd_array: np.ndarray
        nd array from which to compute the hessian matrix.
    sigma: float
        Standard deviation used for the Gaussian kernel to smooth the array. Defaul is 1
    scale: bool
        whether the hessian elements will be scaled by sigma squared. Default is True
    whiteonblack: boolean
        image is white objects on black blackground or not. Default is True

    Return:
    ------------
    list of eigenvalues [eigenvalue1, eigenvalue2, ...]
    """
    return absolute_eigenvaluesh(
        compute_3d_hessian_matrix(nd_array, sigma=sigma, scale=scale, whiteonblack=whiteonblack)
    )

def compute_3d_hessian_matrix(
    nd_array: np.ndarray,
    sigma: float = 1,
    scale: bool = True,
    whiteonblack: bool = True,
) -> np.ndarray:
    """
    Computes the hessian matrix for an nd_array. The implementation was adapted from:
    https://github.com/ellisdg/frangi3d/blob/master/frangi/hessian.py

    Parameters:
    ----------
    nd_array: np.ndarray
        nd array from which to compute the hessian matrix.
    sigma: float
        Standard deviation used for the Gaussian kernel to smooth the array. Defaul is 1
    scale: bool
        whether the hessian elements will be scaled by sigma squared. Default is True
    whiteonblack: boolean
        image is white objects on black blackground or not. Default is True


    Return:
    ----------
    hessian array of shape (..., ndim, ndim)
    """
    ndim = nd_array.ndim

    # smooth the nd_array
    smoothed = ndi.gaussian_filter(nd_array, sigma=sigma, mode="nearest", truncate=3.0)

    # compute the first order gradients
    gradient_list = np.gradient(smoothed)

    # compute the hessian elements
    hessian_elements = [
        np.gradient(gradient_list[ax0], axis=ax1) for ax0, ax1 in combinations_with_replacement(range(ndim), 2)
    ]

    if sigma > 0 and scale:
        # scale the elements of the hessian matrix
        if whiteonblack:
            hessian_elements = [(sigma ** 2) * element for element in hessian_elements]
        else:
            hessian_elements = [-1 * (sigma ** 2) * element for element in hessian_elements]

    # create hessian matrix from hessian elements
    hessian_full = [[()] * ndim for x in range(ndim)]
    # hessian_full = [[None] * ndim] * ndim

    for index, (ax0, ax1) in enumerate(combinations_with_replacement(range(ndim), 2)):
        element = hessian_elements[index]
        hessian_full[ax0][ax1] = element
        if ax0 != ax1:
            hessian_full[ax1][ax0] = element

    hessian_rows = list()
    for row in hessian_full:
        # print(row.shape)
        hessian_rows.append(np.stack(row, axis=-1))

    hessian = np.stack(hessian_rows, axis=-2)
    return hessian