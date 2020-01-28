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

from random import shuffle

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = '/home/stmutasa/Code/Datasets/Scaphoid/'

test_loc_folder = home_dir + 'Test/'
cleanCR_folder = home_dir + 'Cleaned_CR_Single/'

sdl = SDL.SODLoader(data_root=home_dir)
sdd = SDD.SOD_Display()


def pre_proc_localizations(box_dims=1024):

    """
    Pre processes the input for the localization network
    :param box_dims: dimensions of the saved images
    :return:
    """

    group = 'Box_Locs'

    # Load the files and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, cleanCR_folder)
    shuffle(filenames)
    gtboxes = sdl.load_CSV_Dict('filename', 'gtboxes.csv')

    # Global variables
    display, counter, data, data_test, index, pt = [], [0, 0], {}, {}, 0, 0
    heights, weights = [], []

    for file in filenames:

        # Load the Header only first
        try:
            header = sdl.load_DICOM_Header(file, False)
        except Exception as e: continue

        # Retreive the view information
        try:
            view = header['tags'].ViewPosition.upper()
            laterality = header['tags'].ImageLaterality.upper()
            part = header['tags'].BodyPartExamined.upper()
            accno = header['tags'].AccessionNumber
        except Exception as e: continue

        """
            Some odd values that appear:
            view: TAN, LATERAL, NAVICULAR, LLO
            part: PORT WRIST, 
        """

        # Sometimes lateraity is off
        if laterality != 'L' and laterality != 'R':
            if 'RIGHT' in header['tags'].StudyDescription.upper():
                laterality = 'R'
            elif 'LEFT' in header['tags'].StudyDescription.upper():
                laterality = 'L'

        # Skip non wrists or hands
        if 'WRIST' not in part and 'HAND' not in part:
            print('Skipping: ', dst_File)
            continue

        # Set destination filename info
        dst_File = accno + '_' + laterality + '-' + part + '_' + view

        # Now compare to the annotations folder
        try: annotations = gtboxes[dst_File+'.png']['region_shape_attributes']
        except: continue

        # Success, convert the string to a dictionary
        gtbox = json.loads(annotations)
        if not gtbox: continue

        # Success, now load the image and save
        try:
            image, _, _, photometric, _ = sdl.load_DICOM_2D(file)
            if photometric == 1: image *= -1
        except Exception as e:
            # print('DICOM Error: %s' %e)
            continue

        # TODO: Test, display the box
        cn = [gtbox['y'] + (gtbox['height'] // 2), gtbox['x'] + (gtbox['width'] // 2)]
        size = [gtbox['height'], gtbox['width']]
        box_image = sdl.generate_box(image, cn, size, dim3d=False)[0]
        title = ('Dims: %s' %gtbox)
        disp = ('Shape: %sx%s' %(image.shape[0], image.shape[1]))
        sdd.display_single_image(image, False, disp)
        sdd.display_single_image(box_image, False, title=title)

        # Normalize the gtbox, to [ymin, xmin, ymax, xmax, cny, cnx, height, width, origshapey, origshapex]
        shape = image.shape
        gtbox_orig = gtbox
        gtbox = np.asarray([gtbox['y'], gtbox['x'], gtbox['y'] + gtbox['height'], gtbox['x'] + gtbox['width'],
                            gtbox['y']+(gtbox['height']/2), gtbox['x']+(gtbox['width']/2), gtbox['height'], gtbox['width']])
        norm_gtbox = np.asarray([gtbox[0]/shape[0], gtbox[1]/shape[1],
                                 gtbox[2]/shape[0], gtbox[3]/shape[1],
                                 gtbox[4]/shape[0], gtbox[5]/shape[1],
                                 gtbox[6] / shape[0], gtbox[7] / shape[1],
                                 shape[0], shape[1]]).astype(np.float32)

        # Resize and normalize the image
        image = sdl.zoom_2D(image, [box_dims, box_dims])
        image = sdl.adaptive_normalization(image).astype(np.float32)

        # Save the data
        data[index] = {'data': image, 'box_data': norm_gtbox, 'group': group, 'view': dst_File, 'accno': accno}

        # Increment counters
        index += 1
        pt += 1
        heights.append(gtbox[4])
        weights.append(gtbox[5])
        del image

    # Save the data.
    sdl.save_dict_filetypes(data[0])
    sdl.save_segregated_tfrecords(4, data, 'accno', 'data/BOX_LOCS')

    # Done with all patients
    heights, weights = np.asarray(heights), np.asarray(weights)
    print('Made %s bounding boxes. H/W AVG: %s/%s Max: %s/%s' % (
    index, np.average(heights), np.average(weights), np.max(heights), np.max(weights)))


def load_protobuf(training=True):

    """
    Loads the protocol buffer into a form to send to shuffle. No need to oversample for the localization
    """

    # Lambda functions for retreiving our protobuf
    _parse_all = lambda dataset: sdl.load_tfrecords(dataset, [FLAGS.box_dims, FLAGS.box_dims], tf.float32,
                                                    'box_data', tf.float32, [10])

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

        # Large shuffle
        dataset = dataset.shuffle(buffer_size=FLAGS.epoch_size)
        dataset = dataset.repeat(FLAGS.repeats)
        dataset = dataset.map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    else:
        dataset = dataset.map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    scope = 'data_augmentation' if training else 'input'
    with tf.name_scope(scope):
        dataset = dataset.map(DataPreprocessor(training), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch and prefetch
    if training:
        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.batch(FLAGS.batch_size)

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

        # Resize to network size
        image = tf.expand_dims(image, -1)
        image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims], tf.compat.v1.image.ResizeMethod.BICUBIC)

        # Keep track of distortions done
        record['distortion'] = ''

        # Randomly flip
        def flip(mode=None):

            # When flipping, consider that the top left and bottom right corners represent different points now
            ld = record['box_data']

            # For now don't do vertical flip. It doesn't make a ton of sense
            # box_data = [0ymin, 1xmin, 2ymax, 3xmax, 4cny, 5cnx, 6height, 7width, 8origshapey, 9origshapex]
            if mode == 1:

                if FLAGS.net_type == 'BBOX':
                    # img = tf.image.flip_up_down(image) # Vertical flip:
                    # ld0 = (1 - ld[0]) - ld[6] # y=(1-y) - Height to keep top left corner at top left
                    # ld2 = (1 - ld[2]) + ld[6] # y=(1-y) + Height to keep bottom right at bottom right
                    # ld4 = 1 - ld[4] # Flip center point Y-axis
                    # stacked = tf.stack([ld0, ld[1], ld2, ld[3], ld4, ld[5], ld[6], ld[7], ld[8], ld[9]])
                    stacked, img = ld, image

                elif FLAGS.net_type == 'CEN':
                    img = tf.image.flip_up_down(image) # Vertical flip:
                    ld0 = (1 - ld[0]) - ld[6] # y=(1-y) - Height to keep top left corner at top left
                    ld2 = (1 - ld[2]) + ld[6] # y=(1-y) + Height to keep bottom right at bottom right
                    ld4 = 1 - ld[4] # Flip center point Y-axis
                    stacked = tf.stack([ld0, ld[1], ld2, ld[3], ld4, ld[5], ld[6], ld[7], ld[8], ld[9]])

            elif mode == 2:

                img = tf.image.flip_left_right(image) # Horizontal flip
                ld1 = (1 - ld[1]) - ld[7] # x=(1-x) - Width to keep top left corner at top left
                ld3 = (1 - ld[3]) + ld[7] # x=(1-x) + width to keep bottom right at bottom right
                ld5 = 1 - ld[5] # Flip center point X axis
                stacked = tf.stack([ld[0], ld1, ld[2], ld3, ld[4], ld5, ld[6], ld[7], ld[8], ld[9]])

            else: stacked, img = ld, image

            return img, stacked

        # Maxval is not included in the range
        image, record['box_data'] = tf.cond(tf.squeeze(tf.random.uniform([], 0, 2, dtype=tf.int32)) > 0, lambda: flip(1), lambda: flip(0))
        image, record['box_data'] = tf.cond(tf.squeeze(tf.random.uniform([], 0, 2, dtype=tf.int32)) > 0, lambda: flip(2), lambda: flip(0))

        # Random brightness/contrast
        image = tf.image.random_brightness(image, max_delta=2)
        image = tf.image.random_contrast(image, lower=0.995, upper=1.005)

        # Randomly jiggle locations by up to 1.5% of image dims
        rn, lr = [], record['box_data']
        for z in range(6): rn.append(tf.random.uniform([], minval=-0.015, maxval=0.015, dtype=tf.float32))
        record['box_data'] = tf.stack([lr[0]+rn[0], lr[1]+rn[1], lr[2]+rn[2], lr[3]+rn[3], lr[4]+rn[4], lr[5]+rn[5], lr[6], lr[7], lr[8], lr[9]])

        # For noise, first randomly determine how 'noisy' this study will be
        T_noise = tf.random.uniform([], 0, 0.02)

        # Create a poisson noise array
        noise = tf.random.uniform(shape=[FLAGS.network_dims, FLAGS.network_dims, 1], minval=-T_noise, maxval=T_noise)

        # Add the poisson noise
        image = tf.add(image, tf.cast(noise, tf.float32))

        # Make a small dummy image for advanced localization
        record['img_small'] = tf.image.resize_images(image, [FLAGS.network_dims // 8, FLAGS.network_dims // 8],
                                                     tf.compat.v1.image.ResizeMethod.BICUBIC)

    else: # Validation

        # Resize to network size
        image = tf.expand_dims(image, -1)
        image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims], tf.compat.v1.image.ResizeMethod.BICUBIC)
        record['img_small'] = tf.image.resize_images(image, [FLAGS.network_dims//8, FLAGS.network_dims//8], tf.compat.v1.image.ResizeMethod.BICUBIC)

    # Make record image
    record['data'] = image

    return record

#pre_proc_localizations(512)