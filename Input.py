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
import json
import os
import Utils

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
        Generate the anchor boxes here depending on wrist or hand XR
        Interestingly, base it off the original size of the bbox as normalized size varies much more
        """
        if 'WRIST' in part: sh, sw, rat, ratSD, scaSD = 125.8, 122.2, 1.06, 0.232 * 1.25, (19.5 / 125.8 + 24.18 / 122.2) / 2
        else: sh, sw, rat, ratSD, scaSD = 122.3, 127.7, 0.98, 0.196 * 1.25, (17.1 / 122.3 + 24.3 / 127.7) / 2
        anchors = Utils.generate_anchors(image, [sh, sw], 16, ratios=[rat - ratSD, rat, rat + ratSD], scales=[1 - scaSD, 1.0, 1 + scaSD])

        # Generate a GT box measurement in corner format
        ms = image.shape
        gtbox = np.asarray([gtbox['y'], gtbox['x'], gtbox['y'] + gtbox['height'], gtbox['x'] + gtbox['width'],
                            gtbox['y']+(gtbox['height']/2), gtbox['x']+(gtbox['width']/2), gtbox['height'], gtbox['width']])

        # Append IOUs
        IOUs = Utils._iou_calculate(anchors[:, :4], gtbox[:4])
        if np.max(IOUs) <= thresh//2:
            del image, anchors, gtbox
            continue
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
            if an[8] > 0.3 and an[8] < thresh:
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

            # Garbage
            del anchor_box, an, box_data

        # Save q xx patients
        saveq = 110
        if pt % saveq == 0 and pt !=0:
            print('\nMade %s (%s) bounding boxes SO FAR from %s patients. %s Positive and %s Negative (%.6f %%)'
                  % (index, index-lap_count, pt, counter[1], counter[0], 100*counter[1]/index))
            lap_count = index

            if pt < (saveq+5):
                sdl.save_dict_filetypes(data[0])
                sdl.save_tfrecords(data, 1, file_root=('%s/test/BOX_LOCS%s' %(tfrecords_dir, pt//saveq)))
            else: sdl.save_tfrecords(data, 1, file_root=('%s/train/BOX_LOCS%s' %(tfrecords_dir, pt//saveq)))

            del data
            data = {}

        # Increment patient counters
        pt += 1
        del image, anchors, gtbox

    # Save the data.
    sdl.save_tfrecords(data, 1, file_root=('%s/train/BOX_LOCSFin' %tfrecords_dir))

    # Done with all patients
    print('\nMade %s bounding boxes from %s patients. %s Positive and %s Negative (%s %%)'
          %(index, pt, counter[1], counter[0], index/counter[1]))


def load_protobuf(training=True):

    """
    Loads the protocol buffer into a form to send to shuffle. To oversample classes we made some mods...
    Load with parallel interleave -> Prefetch -> Large Shuffle -> Parse labels -> Undersample map -> Flat Map
    -> Prefetch -> Oversample Map -> Flat Map -> Small shuffle -> Prefetch -> Parse images -> Augment -> Prefetch -> Batch
    """

    # Saved the data to [0ymin, 1xmin, 2ymax, 3xmax, cny, cnx, 6height, 7width, 8origshapey, 9origshapex,
    #    10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, 18IOU, 19obj_class, 20#_class]

    # Lambda functions for retreiving our protobuf
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
        _filter_fn = lambda x: sdl.undersample_filter(x['box_data'][19], actual_dists=[0.999, 0.00156], desired_dists=[.9, .1])
        _undersample_filter = lambda x: dataset.filter(_filter_fn)
        _oversample_filter = lambda x: tf.data.Dataset.from_tensors(x).repeat(
            sdl.oversample_class(x['box_data'][19], actual_dists=[0.999, 0.00156], desired_dists=[.9, .1]))

        # Large shuffle, repeat for xx epochs then parse the labels only
        dataset = dataset.shuffle(buffer_size=int(5e5))
        dataset = dataset.repeat(FLAGS.repeats)
        dataset = dataset.map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Now we have the labels, undersample then oversample.
        dataset = dataset.map(_undersample_filter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.flat_map(lambda x: x)
        dataset = dataset.map(_oversample_filter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.flat_map(lambda x: x)

        # Now perform a small shuffle in case we duplicated neighbors, then prefetch before the final map
        dataset = dataset.shuffle(buffer_size=FLAGS.batch_size)

    else: dataset = dataset.map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    scope = 'data_augmentation' if training else 'input'
    with tf.name_scope(scope):
        dataset = dataset.map(DataPreprocessor(training), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch and prefetch
    if training: dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    else: dataset = dataset.batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Make an initializable iterator
    iterator = dataset.make_initializable_iterator()

    # Return data as a dictionary
    return iterator


def load_protobuf_class(training=True):

    """
    Loads the classification network protobuf. No oversampling in this case
    """

    # Saved the data to [0ymin, 1xmin, 2ymax, 3xmax, cny, cnx, 6height, 7width, 8origshapey, 9origshapex,
    #    10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, 18IOU, 19obj_class, 20#_class]

    # Lambda functions for retreiving our protobuf
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

        # Large shuffle, repeat for xx epochs then parse the labels only
        dataset = dataset.shuffle(buffer_size=int(5e5))
        dataset = dataset.repeat(FLAGS.repeats)
        dataset = dataset.map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    else: dataset = dataset.map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    scope = 'data_augmentation' if training else 'input'
    with tf.name_scope(scope):
        dataset = dataset.map(DataPreprocessor(training), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch and prefetch
    if training: dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    else: dataset = dataset.batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

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

        # Image rotation parameters
        angle = tf.random_uniform([], -0.52, 0.52)
        image = tf.contrib.image.rotate(image, angle)

        # Then randomly flip
        image = tf.image.random_flip_left_right(tf.image.random_flip_up_down(image))

        # Random brightness/contrast
        image = tf.image.random_brightness(image, max_delta=2)
        image = tf.image.random_contrast(image, lower=0.995, upper=1.005)

        # For noise, first randomly determine how 'noisy' this study will be
        T_noise = tf.random.uniform([], 0, 0.02)

        # Create a poisson noise array
        noise = tf.random.uniform(shape=[FLAGS.network_dims, FLAGS.network_dims, 1], minval=-T_noise, maxval=T_noise)

        # # Normalize the image
        # image = tf.image.per_image_standardization(image)
        # image = tf.add(image, noise)

        # Add the poisson noise
        image = tf.add(image, tf.cast(noise, tf.float16))

    else: # Validation

        image = tf.expand_dims(image, -1)

        # Normalize the image
        # image = tf.image.per_image_standardization(image)

        # Resize to network size
        image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims], tf.compat.v1.image.ResizeMethod.BICUBIC)

    # Make record image
    record['data'] = image

    return record


#pre_proc_localizations(64, thresh=0.6)