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
    seed = (y, x)  # convert to (row, col)

    if not mask[seed]:
        # optional: avoid flooding background
        return np.zeros_like(mask, dtype=bool)

    seed_mask = np.zeros_like(mask, dtype=bool)
    seed_mask[seed] = True

    return binary_propagation(seed_mask, mask=mask)


def call_if_exists(obj, method):
    if callable(method):
        method()
        return

    if obj and hasattr(obj, method):
        getattr(obj, method)()
    else:
        print(f"[Missing] {method}")


def make_model_callback(widget, method_name):
    def callback(*args):
        call_if_exists(widget.modelFamily, method_name)
    return callback


def make_widget_callback(widget, method):
    def callback(*args):
        call_if_exists(None, method)
    return callback