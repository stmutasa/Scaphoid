"""
Pre processes all of the localization inputs and saves as tfrecords
Also handles loading tfrecords

The standard we're using is saving the top left corner and bottom right corner in numpy format
which is rows x columns or y,x: [ymin, xmin, ymax, xmax]
"""

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import SOD_Display as SDD
from pathlib import Path
import json
import os
import Utils

from random import shuffle

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = '/home/stmutasa/Code/Datasets/Scaphoid/'

test_loc_folder = home_dir + 'Test/'
cleanCR_folder = home_dir + 'Cleaned_CR_Single/'
tfrecords_dir = home_dir + 'tfrecords/'

sdl = SDL.SODLoader(data_root=home_dir)
sdd = SDD.SOD_Display()


def pre_proc_localizations(box_dims=64, thresh=0.6):

    """
    Pre processes the input for the localization network
    :param box_dims: dimensions of the saved images
    :return:
    """

    group = 'Box_Locs'

    # Load the files and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, cleanCR_folder)
    gtboxes = sdl.load_CSV_Dict('filename', 'gtboxes.csv')

    # Global variables
    display, counter, data, lap_count, index, pt = [], [0, 0], {}, 0, 0, 0

    for file in filenames:

        # Load the info
        image, view, laterality, part, accno, header = Utils.filter_DICOM(file)

        # Set destination filename info
        dst_File = accno + '_' + laterality + '-' + part + '_' + view

        # Now compare to the annotations folder
        try: annotations = gtboxes[dst_File + '.png']['region_shape_attributes']
        except: continue

        # Success, convert the string to a dictionary
        gtbox = json.loads(annotations)
        if not gtbox: continue

        # Retreive photometric interpretation (1 = negative XRay) if available
        try:
            photometric = int(header['tags'].PhotometricInterpretation[-1])
            if photometric == 1: image *= -1
        except: continue

        # Normalize image
        image = sdl.adaptive_normalization(image).astype(np.float32)

        """
        Generate the anchor boxes here
        Base; [.118, .153] or [129.4, 127.7], Scales: [0.68, 1.0, 1.3], Ratios [0.82, 1.05, 1.275]
        Feature stride of 16 because that's the stride to get to our feature map size
        """
        anchors = generate_anchors(image, [129.4, 127.7], 16, ratios=[0.82, 1.05, 1.275], scales=[0.68, 1.0, 1.3])

        # Generate a GT box measurement in corner format
        ms = image.shape
        gtbox = np.asarray([gtbox['y'], gtbox['x'], gtbox['y'] + gtbox['height'], gtbox['x'] + gtbox['width'],
                            gtbox['y']+(gtbox['height']/2), gtbox['x']+(gtbox['width']/2), gtbox['height'], gtbox['width']])

        # Append IOUs
        IOUs = _iou_calculate(anchors[:, :4], gtbox[:4])
        if np.max(IOUs) <= thresh:
            print ('Skipping low IOU patient: %s (%s)' %(dst_File, np.max(IOUs)))
        anchors = np.append(anchors, IOUs, axis=1)

        # Normalize the GT boxes
        norm_gtbox = np.asarray([gtbox[0]/ms[0], gtbox[1]/ms[1],
                                 gtbox[2]/ms[0], gtbox[3]/ms[1],
                                 gtbox[4]/ms[0], gtbox[5]/ms[1],
                                 gtbox[6] / ms[0], gtbox[7] / ms[1],
                                 ms[0], ms[1]]).astype(np.float32)

        # Generate boxes by looping through the anchor list
        for an in anchors:

            # Remember anchor = [10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, 18IOU]
            an = an.astype(np.float32)

            # Anchors between IOU of 0.25 and 0.6 do not contribute:
            if an[8] > 0.25 and an[8] < thresh:
                del an
                continue

            # Generate a box at this location
            anchor_box, _ = sdl.generate_box(image, an[4:6].astype(np.int16), an[6:8].astype(np.int16), dim3d=False)

            # Reshape the box to a standard dimension: 128x128
            anchor_box = sdl.zoom_2D(anchor_box, [box_dims, box_dims]).astype(np.float16)

            # Norm the anchor box dimensions
            anchor = [
                an[0]/ms[0], an[1]/ms[1], an[2]/ms[0], an[3]/ms[1], an[4]/ms[0], an[5]/ms[1], an[6]/ms[0], an[7]/ms[1], an[8]
            ]

            # Append the anchor to the norm box
            box_data = np.append(norm_gtbox, anchor)

            # Make object and fracture labels = 1 if IOU > threhsold IOU
            fracture_class = 0
            if box_data[-1] >= thresh: object_class = 1
            else: object_class = 0
            counter[object_class] += 1

            # Append object class and fracture class to box data
            box_data = np.append(box_data, [object_class, fracture_class]).astype(np.float32)

            # Save the data to [0ymin, 1xmin, 2ymax, 3xmax, cny, cnx, 6height, 7width, 8origshapey, 9origshapex,
            #    10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, IOU, obj_class, #_class]
            data[index] = {'data': anchor_box, 'box_data': box_data, 'group': group, 'view': dst_File, 'accno': accno}

            # Increment box count
            index += 1

            # Save copy if this one is positive
            if object_class == 1:
                box_data[14] +=1
                box_data[10] += 1
                data[index] = {'data': anchor_box, 'box_data': box_data, 'group': group, 'view': dst_File, 'accno': accno}
                index += 1

            # Garbage
            del anchor_box, an, box_data

        # Increment patient counters
        pt += 1
        del image

        # Save q xx patients
        saveq = 100
        if pt % saveq == 0:
            print('\nMade %s (%s) bounding boxes SO FAR from %s patients. %s Positive and %s Negative (%.6f %%)'
                  % (index, index-lap_count, pt, counter[1], counter[0], 100*counter[1]/index))
            lap_count = index

            if pt < (saveq+5):
                sdl.save_dict_filetypes(data[0])
                sdl.save_tfrecords(data, 1, file_root=('%s/test/BOX_LOCS%s' %(tfrecords_dir, pt//saveq)))
            else: sdl.save_tfrecords(data, 1, file_root=('%s/train/BOX_LOCS%s' %(tfrecords_dir, pt//saveq)))

            del data
            data = {}

    # Save the data.
    sdl.save_tfrecords(data, 1, file_root=('%s/train/BOX_LOCSFin' %tfrecords_dir))

    # Done with all patients
    print('\nMade %s bounding boxes from %s patients. %s Positive and %s Negative (%s %%)'
          %(index, pt, counter[1], counter[0], index/counter[1]))


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


def load_protobuf(training=True):

    """
    Loads the protocol buffer into a form to send to shuffle. To oversample classes we made some mods...
    Load with parallel interleave -> Prefetch -> Large Shuffle -> Parse labels -> Undersample map -> Flat Map
    -> Prefetch -> Oversample Map -> Flat Map -> Small shuffle -> Prefetch -> Parse images -> Augment -> Prefetch -> Batch
    """

    # Saved the data to [0ymin, 1xmin, 2ymax, 3xmax, cny, cnx, 6height, 7width, 8origshapey, 9origshapex,
    #    10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, 18IOU, 19obj_class, 20#_class]

    # Lambda functions for retreiving our protobuf
    _parse_labels = lambda dataset: sdl.load_tfrecord_labels(dataset, 'box_data', tf.float16, [21])
    _parse_images = lambda dataset: sdl.load_tfrecord_images(dataset, [FLAGS.box_dims, FLAGS.box_dims], tf.float16)
    _parse_all = lambda dataset: sdl.load_tfrecords(dataset, [FLAGS.box_dims, FLAGS.box_dims], tf.float16,
                                                    'box_data', tf.float32, [21])

    # Load tfrecords with parallel interleave if training
    if training:
        filenames = sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir)
        files = tf.data.Dataset.list_files(os.path.join(FLAGS.data_dir, '*.tfrecords'))
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=len(filenames),
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print('******** Loading Files: ', filenames)
    else:
        files = sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir)
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=1)
        print('******** Loading Files: ', files)

    # Shuffle and repeat if training phase
    if training:

        # Define our undersample and oversample filtering functions
        _filter_fn = lambda x: sdl.undersample_filter(x['box_data'][19], actual_dists=[0.999, 0.001], desired_dists=[.7, .3])
        _undersample_filter = lambda x: dataset.filter(_filter_fn)
        _oversample_filter = lambda x: tf.data.Dataset.from_tensors(x).repeat(
            sdl.oversample_class(x['box_data'][19], actual_dists=[0.999, 0.001], desired_dists=[.7, .3]))

        # # Large shuffle, repeat for xx epochs then parse the labels only
        # dataset = dataset.shuffle(buffer_size=FLAGS.epoch_size//20)
        # dataset = dataset.repeat(FLAGS.repeats)
        # dataset = dataset.map(_parse_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #
        # # Now we have the labels, undersample then oversample.
        # # Map allows us to do it in parallel and flat_map's identity function merges the survivors
        # dataset = dataset.map(_undersample_filter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # dataset = dataset.flat_map(lambda x: x)
        # dataset = dataset.map(_oversample_filter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # dataset = dataset.flat_map(lambda x: x)
        #
        # # Now perform a small shuffle in case we duplicated neighbors, then prefetch before the final map
        # dataset = dataset.shuffle(buffer_size=100)
        # dataset = dataset.map(_parse_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        #TODO Testing The above oversample code is not working...
        # dataset = dataset.shuffle(FLAGS.epoch_size).repeat(FLAGS.repeats).map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Large shuffle, repeat for xx epochs then parse the labels only
        dataset = dataset.shuffle(buffer_size=FLAGS.epoch_size)
        dataset = dataset.repeat(FLAGS.repeats)
        dataset = dataset.map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Now we have the labels, undersample then oversample.
        dataset = dataset.map(_undersample_filter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.flat_map(lambda x: x)
        dataset = dataset.map(_oversample_filter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.flat_map(lambda x: x)

        # Now perform a small shuffle in case we duplicated neighbors, then prefetch before the final map
        dataset = dataset.shuffle(buffer_size=100)

    else: dataset = dataset.map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    scope = 'data_augmentation' if training else 'input'
    with tf.name_scope(scope):
        dataset = dataset.map(DataPreprocessor(training), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch and prefetch
    if training: dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    else: dataset = dataset.batch(FLAGS.batch_size)

    # Make an initializable iterator
    iterator = dataset.make_initializable_iterator()

    # Return data as a dictionary
    return iterator


class DataPreprocessor(object):

    # Applies transformations to dataset

  def __init__(self, distords):
    self._distords = distords

  def __call__(self, record):

    """Process img for training or eval."""
    image = record['data']

    if self._distords:  # Training

        # Data Augmentation ------------------ Flip, Contrast, brightness, noise

        # Save the data to [0ymin, 1xmin, 2ymax, 3xmax, cny, cnx, 6height, 7width, 8origshapey, 9origshapex,
        #    10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, 18IOU, 19obj_class, 20#_class]

        # Resize to network size
        image = tf.expand_dims(image, -1)
        image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims], tf.compat.v1.image.ResizeMethod.BICUBIC)

        # Randomly flip
        def flip(mode=None):

            # When flipping, consider that the top left and bottom right corners represent different points now
            ld = record['box_data']

            if mode == 2:

                # Horizontal flip. Flip both the base bounding box and the generated bounding box
                img = tf.image.flip_left_right(image) # Horizontal flip
                ld1 = (1 - ld[1]) - ld[7] # x=(1-x) - Width to keep top left corner at top left
                ld3 = (1 - ld[3]) + ld[7] # x=(1-x) + width to keep bottom right at bottom right
                ld5 = 1 - ld[5] # Flip center point X axis
                ld11 = (1 - ld[11]) - ld[17]  # x=(1-x) - Width to keep top left corner at top left
                ld13 = (1 - ld[13]) + ld[17]  # x=(1-x) + width to keep bottom right at bottom right
                ld15 = 1 - ld[15]  # Flip center point X axis
                stacked = tf.stack([ld[0], ld1, ld[2], ld3, ld[4], ld5, ld[6], ld[7], ld[8], ld[9],
                                    ld[10], ld11, ld[12], ld13, ld[14], ld15, ld[16], ld[17], ld[18], ld[19],
                                    ld[20]])

            else: stacked, img = ld, image

            return img, stacked

        # Maxval is not included in the range
        image, record['box_data'] = tf.cond(tf.squeeze(tf.random.uniform([], 0, 2, dtype=tf.int32)) > 0, lambda: flip(2), lambda: flip(0))

        # Change IOU cutoff function
        def cutoff2(mode=None):

            # When flipping, consider that the top left and bottom right corners represent different points now
            ld = record['box_data']

            if mode == 2:

                # Horizontal flip. Flip both the base bounding box and the generated bounding box
                ld19 = 1
                stacked = tf.stack([ld[0], ld[1], ld[2], ld[3], ld[4], ld[5], ld[6], ld[7], ld[8], ld[9],
                                    ld[10], ld[11], ld[12], ld[13], ld[14], ld[15], ld[16], ld[17], ld[18], ld19,
                                    ld[20]])

            else:
                ld19 = 0
                stacked = tf.stack([ld[0], ld[1], ld[2], ld[3], ld[4], ld[5], ld[6], ld[7], ld[8], ld[9],
                                    ld[10], ld[11], ld[12], ld[13], ld[14], ld[15], ld[16], ld[17], ld[18], ld19,
                                    ld[20]])

            return stacked

        # # Change IOU cutoff. Note that we oversampled the tfrecord cutoff, not this one...
        # cutoff = 0.3
        # record['box_data'] = tf.cond(record['box_data'][18] > cutoff, lambda: cutoff2(2), lambda: cutoff2(0))

        # Random brightness/contrast
        image = tf.image.random_brightness(image, max_delta=2)
        image = tf.image.random_contrast(image, lower=0.995, upper=1.005)

        # # Randomly jiggle locations by up to 1.5% of image dims
        # rn, lr = [], record['box_data']
        # for z in range(6): rn.append(tf.random.uniform([], minval=-0.015, maxval=0.015, dtype=tf.float32))
        # record['box_data'] = tf.stack([lr[0]+rn[0], lr[1]+rn[1], lr[2]+rn[2], lr[3]+rn[3], lr[4]+rn[4], lr[5]+rn[5], lr[6], lr[7], lr[8], lr[9]])

        # For noise, first randomly determine how 'noisy' this study will be
        T_noise = tf.random.uniform([], 0, 0.02)

        # Create a poisson noise array
        noise = tf.random.uniform(shape=[FLAGS.network_dims, FLAGS.network_dims, 1], minval=-T_noise, maxval=T_noise)

        # Add the poisson noise
        image = tf.add(image, tf.cast(noise, tf.float16))

    else: # Validation

        # Resize to network size
        image = tf.expand_dims(image, -1)
        image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims], tf.compat.v1.image.ResizeMethod.BICUBIC)

    # Make record image
    record['data'] = image

    return record


#pre_proc_localizations(64, thresh=0.6)