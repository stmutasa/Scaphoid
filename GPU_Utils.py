"""
Utility functions that run on CuPy instead of numpy
"""

#import cupy as np
import numpy as np


def generate_box(image, origin=[], size=32, display=False, dim3d=True, z_overwrite=None):
    """
    This function returns a cube from the source image
    :param image: The source image
    :param origin: Center of the cube as a matrix of x,y,z [z, y, x]
    :param size: dimensions of the cube in mm
    :param dim3d: Whether this is 3D or 2D
    :param z_overwrite: Use this to overwrite the size of the Z axis, otherwise it defaults to half
    :return: cube: the cube itself
    """

    # Sometimes size is an array
    try:
        sizey = int(size[0])
        sizex = int(size[1])
    except:
        sizey = size
        sizex = size

    # First implement the 2D version
    if not dim3d:

        # Make the starting point = center-size unless near the edge then make it 0
        startx = max(origin[1] - sizex / 2, 0)
        starty = max(origin[0] - sizey / 2, 0)

        # If near the far edge, make it fit inside the image
        if (startx + sizex) > image.shape[1]:
            startx = image.shape[1] - sizex
        if (starty + sizey) > image.shape[0]:
            starty = image.shape[0] - sizey

        # Convert to integers
        startx = int(startx)
        starty = int(starty)

        # Now retreive the box
        box = image[starty:starty + sizey, startx:startx + sizex]

        # If boxes had to be shifted, we have to calculate a new 'center' of the nodule in the box
        new_center = [int(sizey / 2 - ((starty + sizey / 2) - origin[0])),
                      int(sizex / 2 - ((startx + sizex / 2) - origin[1]))]

        return box, new_center

    # first scale the z axis in half
    if z_overwrite:
        sizez = z_overwrite
    else:
        sizez = int(size / 2)

    # Make the starting point = center-size unless near the edge then make it 0
    startx = max(origin[2] - size / 2, 0)
    starty = max(origin[1] - size / 2, 0)
    startz = max(origin[0] - sizez / 2, 0)

    # If near the far edge, make it fit inside the image
    if (startx + size) > image.shape[2]:
        startx = image.shape[2] - size
    if (starty + size) > image.shape[1]:
        starty = image.shape[1] - size
    if (startz + sizez) > image.shape[0]:
        startz = image.shape[0] - sizez

    # Convert to integers
    startx = int(startx)
    starty = int(starty)
    startz = int(startz)

    # Now retreive the box
    box = image[startz:startz + sizez, starty:starty + size, startx:startx + size]

    # If boxes had to be shifted, we have to calculate a new 'center' of the nodule in the box
    new_center = [int(sizez / 2 - ((startz + sizez / 2) - origin[0])),
                  int(size / 2 - ((starty + size / 2) - origin[1])),
                  int(size / 2 - ((startx + size / 2) - origin[2]))]

    # display if wanted
    if display: print(image.shape, startz, starty, startx, box.shape, 'New Center:', new_center)

    return box, new_center

def _iou_calculate(boxes1, boxes2):

    """
    Calculates the IOU of two boxes
    :param boxes1: [n, 4] [ymin, xmin, ymax, xmax]
    :param boxes2: [n, 4]
    :return: Overlaps of each box pair (aka DICE score)
    """

    # Split the coordinates
    ymin_1, xmin_1, ymax_1, xmax_1 = np.split(boxes1, indices_or_sections=len(boxes1[1]), axis=1)  # ymin_1 shape is [N, 1]..
    ymin_2, xmin_2, ymax_2, xmax_2 = np.split(boxes2, indices_or_sections=len(boxes2), axis=0)  # ymin_2 shape is [M, ]..

    # Retreive any overlaps of the corner points of the box
    max_xmin, max_ymin = np.maximum(xmin_1, xmin_2), np.maximum(ymin_1, ymin_2)
    min_xmax, min_ymax = np.minimum(xmax_1, xmax_2), np.minimum(ymax_1, ymax_2)

    # Retreive overlap along each dimension: Basically if the upper right corner of one box is above the lower left of another, there is overlap
    overlap_h = np.maximum(0., min_ymax - max_ymin)  # avoid h < 0
    overlap_w = np.maximum(0., min_xmax - max_xmin)

    # Cannot overlap if one of the dimension overlaps is 0
    overlaps = overlap_h * overlap_w

    # Calculate the area of each box
    area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)  # [N, 1]
    area_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)  # [M, ]

    # Calculate overlap (intersection) over union like Dice score. Union is just the areas added minus the overlap
    iou = overlaps / (area_1 + area_2 - overlaps)

    return iou


def generate_anchors(image, base_anchor_size, feature_stride, ratios, scales):

    """
    For generating anchor boxes
    :param image: The input image over which we generate anchors
    :param base_anchor_size: The base anchor size for this feature map
    :param feature_stride: int, stride of feature map relative to the image in pixels
    :param ratios: [1D array] of anchor ratios of width/height. i.e [0.5, 1, 2]
    :param scales: [1D array] of anchor scales in original space
    :return:
    """

    # Make Numpy Arrays
    if type(base_anchor_size) is not np.ndarray: base_anchor_size = np.asarray(base_anchor_size)
    if type(ratios) is not np.ndarray: ratios = np.asarray(ratios)
    if type(scales) is not np.ndarray: scales = np.asarray(scales)

    # Get shape
    shapey, shapex = image.shape

    # Generate a base anchor
    base_anchor = np.array([0, 0, base_anchor_size[0], base_anchor_size[1]], np.float32)
    base_anchors = _enum_ratios(_enum_scales(base_anchor, scales), ratios)
    _, _, ws, hs = np.split(base_anchors, indices_or_sections=len(base_anchors[1]), axis=1)

    # Create sequence of numbers
    y_centers = np.arange(0, shapey, feature_stride)
    x_centers = np.arange(0, shapex, feature_stride)

    # Broadcast parameters to a grid of x and y coordinates
    ws, hs = np.squeeze(ws), np.squeeze(hs)
    x_centers, y_centers = np.meshgrid(x_centers, y_centers)
    ws, x_centers = np.meshgrid(ws, x_centers)
    hs, y_centers = np.meshgrid(hs, y_centers)

    # Stack anchor centers and box sizes. Reshape to get a list of (x, y) and a list of (h, w)
    anchor_centers = np.reshape(np.stack([x_centers, y_centers], 2), [-1, 2])
    box_sizes = np.reshape(np.stack([ws, hs], axis=2), [-1, 2])

    # Convert to corner coordinates
    anchors = np.concatenate([anchor_centers - 0.5 * box_sizes, anchor_centers + 0.5 * box_sizes], axis=1)

    # Append all to one array with anchor [corner coordinates, centers, box_sizes]
    all = np.append(anchors, np.append(anchor_centers, box_sizes, axis=1), axis=1)

    # Filter outside anchors
    filtered = _filter_outside_anchors(anchors=all, img_dim=image.shape)

    return filtered


def _enum_scales(base_anchor, anchor_scales):

    '''
    :param base_anchor: [y_center, x_center, h, w]
    :param anchor_scales: different scales, like [0.5, 1., 2.0]
    :return: return base anchors in different scales.
            Example:[[0, 0, 128, 128],[0, 0, 256, 256],[0, 0, 512, 512]]
    '''

    anchor_scales = base_anchor * np.reshape(np.asarray(anchor_scales, dtype=np.float32), newshape=(len(anchor_scales), 1))
    return anchor_scales


def _enum_ratios(anchors, anchor_ratios):

    '''
    :param anchors: base anchors in different scales
    :param anchor_ratios:  ratio = h / w
    :return: base anchors in different scales and ratios
    '''

    # Unstack along the vertical dimension
    _, _, hs, ws = np.split(anchors, indices_or_sections=len(anchors[1]), axis=1)

    # Calculate squares of the anchor ratios
    sqrt_ratios = np.sqrt(anchor_ratios)
    sqrt_ratios = np.transpose(np.expand_dims(sqrt_ratios, axis=1))

    # Reshape the anchors
    ws = np.reshape(ws / sqrt_ratios, [-1])
    hs = np.reshape(hs * sqrt_ratios, [-1])

    num_anchors_per_location = ws.shape[0]

    ratios = np.transpose(np.stack([np.zeros([num_anchors_per_location, ]), np.zeros([num_anchors_per_location, ]), ws, hs]))

    return ratios


def _filter_outside_anchors(anchors, img_dim):

    """
    Removes anchor proposals with values outside the image
    :param anchors: The anchor proposals [xmin, ymin, xmax, ymax]
    :param img_dim: image dimensions (assumes square input)
    :return: the indices of the anchors not outside the image boundary
    """

    # Unpack the rank R tensor into multiple rank R-1 tensors along axis
    ymin, xmin, ymax, xmax, cny, cnx, wh, ww = np.split(anchors, indices_or_sections=len(anchors[1]), axis=1)

    # Return True for indices inside the image
    xmin_index, ymin_index = np.greater_equal(xmin, 0), np.greater_equal(ymin, 0)
    xmax_index, ymax_index = np.less_equal(xmax, img_dim[1]), np.less_equal(ymax, img_dim[0])

    # Now clean up the indices and return them
    indices = np.transpose(np.squeeze(np.stack([ymin_index, xmin_index, ymax_index, xmax_index])))
    indices = indices.astype(np.float32)
    indices = np.sum(indices, axis=1)
    indices = np.where(np.equal(indices, 4.0))

    # Gather the valid anchors
    valid_anchors = np.squeeze(np.take(anchors, indices, axis=0))

    return valid_anchors