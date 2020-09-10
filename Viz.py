"""
Graphs and visualizations for the neural network
"""

import numpy as np
import numpy as cp
import tensorflow as tf
import cv2
import time

import Utils
import SODLoader as SDL
import SOD_Display as SDD
import GPU_Utils as gut
import LocModel as network

from random import shuffle
from math import sqrt
from functools import reduce
import matplotlib.pyplot as plt

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = '/home/stmutasa/Code/Datasets/Scaphoid/'
tfrecords_dir = home_dir + 'tfrecords/temp1/'

test_loc_folder = home_dir + 'Test/'
cleanCR_folder = home_dir + 'Cleaned_CR/A2/'
hedgeCR_folder = home_dir + 'Cleaned_CR_Hedge/'
label_folder = home_dir + 'Labels/'

sdl = SDL.SODLoader(data_root=home_dir)
sdd = SDD.SOD_Display()

# Define some of the data variables
tf.app.flags.DEFINE_string('data_dir', tfrecords_dir, """Path to the data directory.""")
tf.app.flags.DEFINE_string('training_dir', 'training/', """Path to the training directory.""")
tf.app.flags.DEFINE_integer('box_dims', 64, """dimensions to save files""")
tf.app.flags.DEFINE_integer('network_dims', 64, """dimensions of the network input""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.75, """ Keep probability""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory where to retrieve checkpoint files""")
tf.app.flags.DEFINE_string('net_type', 'RPNC', """Network predicting CEN or BBOX""")
tf.app.flags.DEFINE_string('RunInfo', 'RPN_FL6/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 0, """Which GPU to use""")


def execute_Test():

    """
    loads the raw test images. Generates anchors, Runs anchors, Filters anchors, Saves outputs
    :return:
    """

    # Load the labels and files and randomly shuffle them
    labels = sdl.load_CSV_Dict('Accno', path=label_folder + 'Test_Lbls.csv')
    labels.update(sdl.load_CSV_Dict('Accno', path=label_folder + 'Test_Lbls_EZ.csv'))
    scores = sdl.load_CSV_Dict('Accno', 'best_run.csv')
    filenames = sdl.retreive_filelist('dcm', True, cleanCR_folder) + sdl.retreive_filelist('dcm', True, hedgeCR_folder)
    shuffle(filenames)
    totimg = len(filenames)
    print ('Found %s image files and %s test labels, ...starting' %(totimg, len(labels)))

    # Global variables
    data, index, procd, counter =  {}, 0, 0, [0, 0]
    tot_props = 0

    for file in filenames:

        # Skip if the this is NOT a test file
        test_acc = file.split('/')[-2]
        try:
            _ = labels[test_acc]
            pass
        except: continue

        # Save protobuff and get epoch size
        try: epoch_size, ID, cls = pre_proc_localizations(64, file, labels, group='test')
        except: continue

        # Get factors of epoch size for batch size and return number closest to 1k
        ep_factors = factors(epoch_size)
        batch_size = min(ep_factors, key=lambda x: abs(x - 3000))
        if batch_size < 100 or batch_size > 6000: batch_size = 100

        # Load this patient
        iterator = load_protobuf(batch_size)

        # Run the patient through
        result_dict, index = inference(iterator, epoch_size, batch_size, index)

        # Keep only the top x proposals from the dict
        top_n = 5
        if len(result_dict) >= top_n:

            # Sort items by obj_prob in iterated list. Use reverse to get biggest first, take n with slicing then make dict
            result_dict = dict(sorted(result_dict.items(), key=lambda x: x[1]['obj_prob'], reverse=True)[:top_n])

        # pertinent Box data: 10yamin, 11xamin, 12yamax, 13xamax, 8
        if len(result_dict) == 0: continue

        # Get the original image
        try:
            image, _, _, photometric, _ = sdl.load_DICOM_2D(file)
            if photometric == 1: image *= -1
        except: continue

        # Use the score as the overlay value
        accno = ID.split('_')[0]
        try:
            if cls == 1: score = scores[accno]['Calc_Prob']
            else: score = scores[accno]['Class_1_Probability']
        except: continue

        # Normalize
        image = sdl.adaptive_normalization(image).astype(np.float32)

        # reconvert to int16
        # image = convert_high_bit_image(image, -32000, 32000, np.int16)
        image = sdl.normalize_dynamic_range(image, -32768, 32767, dtype=np.int16, dim3d=False)
        # image = sdl.normalize_dynamic_range(image, 0, 255, dtype=np.uint8, dim3d=False)

        # Get the box dimensions
        boxes = np.zeros([len(result_dict), 4], np.float32)

        # Index for the boxes we will copy
        _idx = 0

        # Loop through and populate the boxes variable with the dimensions
        for _, _dict in result_dict.items():

            # Get the image shape
            ys, xs = image.shape

            # convert from 10ymin, 11xamin, 12yamax, 13xamax to xmin, ymin, width, height Un-Normalize while doing it
            xmin = _dict['box_data'][11] * xs
            ymin = _dict['box_data'][10] * ys
            xmax = _dict['box_data'][13] * xs
            ymax = _dict['box_data'][12] * ys
            height, width = ymax - ymin, xmax - xmin

            # Append to boxes
            boxes[_idx] = [xmin, ymin, width, height]
            _idx +=1

        # Overaly the boxes on the original image
        if 'PA' in ID: overlay = sdd.draw_box_cv(image, boxes, 'Fracture Probability =%.2f (%s)' %(float(score), cls))
        else: overlay = sdd.draw_box_cv(image, boxes, '')

        # Display to test
        #sdd.display_single_image(overlay, True)

        # Save the image, RGB
        savefile = 'data/Viz/' + ID + '.png'
        sdd.save_image(overlay, savefile)

        # Update count
        counter[cls] += len(result_dict)

        # Display
        print ('\n *** Made %s Test boxes of the scaphoid from %s proposals in image %s (IMG %s of %s, Objects so far: %s) *** \n'
               %(len(result_dict), epoch_size, ID, procd, totimg, counter))

        # Garbage and tracking
        procd += 1
        tot_props += epoch_size
        del result_dict, iterator, overlay, boxes, image


def factors(n):
        step = 2 if n%2 else 1
        return set(reduce(list.__add__,
                    ([i, n//i] for i in range(1, int(sqrt(n))+1, step) if n % i == 0)))


def convert_high_bit_image(img, target_type_min, target_type_max, target_type):
    """
    Converts a higher bit image to a lower bit one
    :param img: input image
    :param target_type_min: minimum target value
    :param target_type_max: maximum target value
    :param target_type: numpy data type of target
    :return: The converted image
    """
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def pre_proc_localizations(box_dims, file, labels, group='proposals'):

    """
    Pre processes the input for the classification
    :param box_dims: dimensions of the saved images
    :param file: the patient to work on
    :return:
    """

    # Global variables
    display, counter, data, index = [], [0, 0], {}, 0

    # Load the info
    image, view, laterality, part, accno, header = Utils.filter_DICOM(file, show_debug=True)

    # Skip laterals
    if 'LAT' in view:
        return

    # Set destination filename info
    dst_File = accno + '_' + laterality + '-' + part + '_' + view

    # Get the label
    try: fracture_class = int(labels[accno]['Lbl'])
    except:
        print('Error: Cant find label for ', dst_File)
        return

    # Normalize image
    image = sdl.adaptive_normalization(image).astype(np.float32)

    # Shuttle image to GPU
    #image = cp.asarray(image)

    """
        Generate the anchor boxes here depending on wrist or hand XR
        Interestingly, base it off the original size of the bbox as normalized size varies much more
    """
    if 'WRIST' in part:
        sh, sw, rat, ratSD, scaSD = 125.8, 122.2, 1.06, 0.232 * 1.25, (19.5 / 125.8 + 24.18 / 122.2) / 2
    else:
        sh, sw, rat, ratSD, scaSD = 122.3, 127.7, 0.98, 0.196 * 1.25, (17.1 / 122.3 + 24.3 / 127.7) / 2

    anchors = Utils.generate_anchors(image, [sh, sw], 10, ratios=[rat - ratSD, rat, rat + ratSD],
                                     scales=[1 - scaSD, 1.0, 1 + scaSD])

    # Generate a dummy GT Box
    ms = image.shape
    gtbox = cp.asarray([1, 1, 1, 1, 1, 1, 1, 1])

    # Append dummy zero IOUs
    IOUs = cp.expand_dims(cp.zeros_like(anchors[:, 1]), -1)
    anchors = np.append(anchors, IOUs, axis=1)

    # Normalize the GT boxes
    norm_gtbox = cp.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ms[0], ms[1]]).astype(cp.float32)

    # Generate boxes by looping through the anchor list
    for an in anchors:

        # Remember anchor = [10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, 18IOU]
        an = an.astype(cp.float32)

        # Generate a box at this location
        anchor_box, _ = gut.generate_box(image, an[4:6].astype(cp.int16), an[6:8].astype(cp.int16), dim3d=False)

        # Reshape the box to a standard dimension: 128x128
        anchor_box = sdl.zoom_2D(anchor_box, [box_dims, box_dims]).astype(cp.float16)

        # Norm the anchor box dimensions
        anchor = [
            an[0]/ms[0], an[1]/ms[1], an[2]/ms[0], an[3]/ms[1], an[4]/ms[0], an[5]/ms[1], an[6]/ms[0], an[7]/ms[1], an[8]
        ]

        # Append the anchor to the norm box
        box_data = cp.append(norm_gtbox, anchor)

        # Make object and fracture labels
        object_class = 0

        # Append object class and fracture class to box data
        box_data = cp.append(box_data, [object_class, fracture_class]).astype(cp.float32)

        # Save the data to [0ymin, 1xmin, 2ymax, 3xmax, cny, cnx, 6height, 7width, 8origshapey, 9origshapex,
        #    10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, 18IOU, 19obj_class, 20#_class]
        # data[index] = {'data': cp.asnumpy(anchor_box), 'box_data': cp.asnumpy(box_data), 'group': group, 'view': dst_File, 'accno': accno}
        data[index] = {'data': anchor_box, 'box_data': box_data, 'group': group, 'view': dst_File, 'accno': accno}

        # Increment box count
        index += 1

        # Garbage
        del anchor_box, an, box_data

    # Increment patient counters
    del image

    sdl.save_dict_filetypes(data[0], (tfrecords_dir + 'filetypes_tmp'))
    sdl.save_tfrecords(data, 1, file_root=('%s/PROPS' %tfrecords_dir))

    del data
    return index, dst_File, fracture_class

def load_protobuf(batch_size):

    """
    Loads the protocol buffer
    """

    # Saved the data to [0ymin, 1xmin, 2ymax, 3xmax, cny, cnx, 6height, 7width, 8origshapey, 9origshapex,
    #    10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, 18IOU, 19obj_class, 20#_class]

    # Lambda functions for retreiving our protobuf
    _parse_all = lambda dataset: sdl.load_tfrecords(dataset, [FLAGS.box_dims, FLAGS.box_dims], tf.float16,
                                                    'box_data', tf.float32, [21], pickle_dict=(tfrecords_dir + 'filetypes_tmp_pickle.p'))

    # Load tfrecords
    files = sdl.retreive_filelist('tfrecords', False, path=tfrecords_dir)
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=1)

    # Parse and preprocess the dataset
    dataset = dataset.map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    with tf.name_scope('input'):
        dataset = dataset.map(DataPreprocessor(False), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

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
    image = tf.expand_dims(image, -1)

    # # Normalize the image
    # image = tf.image.per_image_standardization(image)

    # Resize to network size
    image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims], tf.compat.v1.image.ResizeMethod.BICUBIC)

    # Make record image
    record['data'] = image

    return record


def inference(iterator, epoch_size, batch_size, index):

    # Run default graph on GPU 1 always
    with tf.device('/gpu:' + str(FLAGS.GPU)):

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Load the images and labels.
        data = iterator.get_next()

        # Define input shape
        data['data'] = tf.reshape(data['data'], [batch_size, FLAGS.network_dims, FLAGS.network_dims, 1])

        # Perform the forward pass:
        all_logits = network.forward_pass_RPN(data['data'], phase_train=phase_train)

        # Labels and logits
        softmax = tf.nn.softmax(all_logits[0])

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=1)

        # Allow memory placement growth
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as mon_sess:

            # Retreive the checkpoint
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir+FLAGS.RunInfo)

            # Restore the model
            mon_sess.run([var_init, iterator.initializer])
            saver.restore(mon_sess, ckpt.model_checkpoint_path)

            # Initialize the step counter
            step, made = 0, False

            # Set the max step count
            max_steps = int(epoch_size / batch_size)

            # Dict to save
            save_dict = {}

            try:
                while step < max_steps:

                    """
                    Our goal is to take all the positive indices predicted and save all of the data for classification
                    Don't use anything that requires knowlege of the actual label
                    Maybe save only the 10 most positive examples?? Dunno, bigly.
                    """

                    # Load what we need
                    _softmax, _data = mon_sess.run([softmax, data], feed_dict={phase_train: False})

                    # Only keep positive
                    class_preds = np.argmax(_softmax, axis=1)
                    positives_here = np.sum(class_preds)
                    if positives_here == 0:
                        del _softmax, _data
                        step += 1
                        continue

                    # Retreive the indices
                    pos_indices = np.where(np.equal(class_preds, 1))[0]

                    # Take
                    for idx in pos_indices:
                        save_dict[index] = {'data': _data['data'][idx].astype(np.float16),
                                        'box_data': _data['box_data'][idx].astype(np.float16),
                                        'group': _data['group'][idx].decode('utf-8'),
                                        'view': _data['view'][idx].decode('utf-8'),
                                        'accno': _data['accno'][idx].decode('utf-8'),
                                        'obj_prob':_softmax[idx, 1]}
                        index +=1

                    # Increment step
                    del _softmax, _data, class_preds
                    step += 1

            except tf.errors.OutOfRangeError:
                print('Done with Training - Epoch limit reached')

            finally:

                # Shut down the session
                mon_sess.close()

        tf.reset_default_graph()
        return save_dict, index


def main(argv=None):
    # execute()
    # time.sleep(25000)
    execute_Test()
    #execute_hedge()

if __name__ == '__main__':
    tf.app.run()