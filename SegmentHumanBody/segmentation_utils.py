from collections import deque
import numpy as np


def get_slice_accessor_dimension(volumeNode):
    npArray = np.zeros((3, 3))
    volumeNode.GetIJKToRASDirections(npArray)
    firstAxis = np.transpose(npArray)[0]
    return int(np.argmax(np.abs(firstAxis)))


def get_volume_slice(volume, accessor_dim, slice_index):
    if accessor_dim == 0:
        return volume[slice_index, :, :]
    elif accessor_dim == 1:
        return volume[:, slice_index, :]
    else:
        return volume[:, :, slice_index]


def get_annotation_slice(segmentationArray, accessor_dim, slice_index):
    if accessor_dim == 0:
        return segmentationArray[slice_index, :, :]
    elif accessor_dim == 1:
        return segmentationArray[:, slice_index, :]
    else:
        return segmentationArray[:, :, slice_index]


def set_volume_slice(volumeMask, accessor_dim, slice_index, sliceMask):
    if accessor_dim == 0:
        volumeMask[slice_index, :, :] = sliceMask
    elif accessor_dim == 1:
        volumeMask[:, slice_index, :] = sliceMask
    else:
        volumeMask[:, :, slice_index] = sliceMask


def combine_multiple_masks(masks):
    if len(masks) == 0:
        return None

    first = masks[0]
    if first.ndim > 2:
        finalMask = np.zeros_like(first[0], dtype=bool)
        for mask in masks:
            finalMask |= mask[0].astype(bool)
        return finalMask

    finalMask = np.zeros_like(first, dtype=bool)
    for mask in masks:
        finalMask |= mask.astype(bool)
    return finalMask


def bfs_connected_component(mask, promptPointXY):
    # promptPointXY is [x, y], convert to [row, col]
    start = [promptPointXY[1], promptPointXY[0]]

    rows, cols = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    targetValue = mask[start[0], start[1]]

    q = deque([start])
    visited[start[0], start[1]] = True

    while q:
        r, c = q.popleft()

        neighbors = [
            (r + 1, c),
            (r - 1, c),
            (r, c + 1),
            (r, c - 1),
        ]

        for rr, cc in neighbors:
            if 0 <= rr < rows and 0 <= cc < cols:
                if not visited[rr, cc] and mask[rr, cc] == targetValue:
                    visited[rr, cc] = True
                    q.append((rr, cc))

    return visited