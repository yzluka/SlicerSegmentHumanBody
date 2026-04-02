from scipy.ndimage import binary_propagation
import numpy as np

def extract_connected_component(mask: np.ndarray, point_xy):
    """
    Extract connected component from a binary mask given a seed point.

    Parameters:
        mask: 2D boolean numpy array
        point_xy: (x, y) coordinate

    Returns:
        Boolean mask of the connected component
    """
    x, y = point_xy
    seed = (y, x)  

    if not mask[seed]:
        return np.zeros_like(mask, dtype=bool)

    seed_mask = np.zeros_like(mask, dtype=bool)
    seed_mask[seed] = True

    return binary_propagation(seed_mask, mask=mask)


def call_if_exists(obj, method_name, *args, **kwargs):
    if obj and hasattr(obj, method_name):
        return getattr(obj, method_name)(*args, **kwargs)


def get_slice_from_volume(volume_array, axis, slice_index):
    
    if axis == 2:
        return volume_array[:, :, slice_index]
    elif axis == 1:
        return volume_array[:, slice_index, :]
    else:
        return volume_array[slice_index, :, :]


def write_slice_to_volume(target_array, slice_data, axis, slice_index):
    
    if axis == 2:
        target_array[:, :, slice_index] = slice_data
    elif axis == 1:
        target_array[:, slice_index, :] = slice_data
    else:
        target_array[slice_index, :, :] = slice_data