""" Evaluating the localization network on a CPU or GPU """

import os
import time

import LocModel as network
import tensorflow as tf
import SODTester as SDT
import SODLoader as SDL
import SOD_Display as SDD
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sdl= SDL.SODLoader('/home/stmutasa/Code/Datasets/Scaphoid/')
sdd = SDD.SOD_Display()

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the data variables
tf.app.flags.DEFINE_string('data_dir', 'data/test/', """Path to the data directory.""")
tf.app.flags.DEFINE_string('training_dir', 'testing/', """Path to the training directory.""")
tf.app.flags.DEFINE_integer('box_dims', 1024, """dimensions to save files""")
tf.app.flags.DEFINE_integer('network_dims', 512, """dimensions of the network input""")
tf.app.flags.DEFINE_integer('epoch_size', 200, """How many examples""")
tf.app.flags.DEFINE_integer('batch_size', 200, """Number of images to process in a batch.""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ Keep probability""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('net_type', 'CEN', """Network predicting CEN or BBOX""")
tf.app.flags.DEFINE_string('RunInfo', 'Center/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 0, """Which GPU to use""")

# Define a custom training class
def test():


    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Load the images and labels.
        iterator = network.inputs(training=False, skip=True)
        data = iterator.get_next()

        # Define input shape
        data['data'] = tf.reshape(data['data'], [FLAGS.batch_size, FLAGS.network_dims, FLAGS.network_dims])

        # Perform the forward pass:
        if FLAGS.net_type == 'BBOX':
            logits = network.forward_pass((data['data'], data['img_small']), phase_train=phase_train)
            labels = data['box_data'][:, :4]
        elif FLAGS.net_type == 'CEN':
            logits = network.forward_pass_center((data['data'], data['img_small']), phase_train=phase_train)
            labels = data['box_data'][:, 4:6]

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=1)

        # Tester instance
        sdt = SDT.SODTester(False, True)

        # Allow memory placement growth
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as mon_sess:

            # Print run info
            print("*** Output Run %s on GPU %s ****" % (FLAGS.RunInfo, FLAGS.GPU))

            # Retreive the checkpoint
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir+FLAGS.RunInfo)

            # Initialize the variables
            mon_sess.run(var_init)
            mon_sess.run(iterator.initializer)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(mon_sess, ckpt.model_checkpoint_path)

            else:
                print ('No checkpoint file found')

            # Run inference
            _logs, _data = mon_sess.run([logits, data], feed_dict={phase_train: False})

            # Display the predictions
            if FLAGS.net_type == 'BBOX': display = bbox_display(_data, _logs)
            elif FLAGS.net_type == 'CEN': display = cen_display(_data, _logs)

            # Display
            sdd.display_volume(display, True)

            # Shut down the session
            mon_sess.close()



def bbox_display(_data, _logs):

    """
    Displays the bounding boxes made by a bbox predictor type network
    :return:
    """

    # Trackers
    show_all = []

    # Output box_data = [0ymin, 1xmin, 2ymax, 3xmax, 4cny, 5cnx, 6height, 7width, 8origshapey, 9origshapex]
    # for z in range (FLAGS.epoch_size):
    for z in range(FLAGS.epoch_size):

        # Remake GTBox from saved data
        bd = _data['box_data'][z]
        width, height = (bd[3] - bd[1]) * bd[9], (bd[2] - bd[0]) * bd[8]
        xmin, ymin = bd[1] * bd[9], bd[0] * bd[8]
        gtbox_recon = {'name': 'rect', 'x': xmin, 'y': ymin, 'width': width, 'height': height}

        # Get parameters for center box
        cn = [gtbox_recon['y'] + (gtbox_recon['height'] // 2),
              gtbox_recon['x'] + (gtbox_recon['width'] // 2)]
        size = [gtbox_recon['height'], gtbox_recon['width']]
        _image = sdl.zoom_2D(_data['data'][z], [bd[8], bd[9]])
        box_true = sdl.generate_box(_image, cn, size, dim3d=False)[0]

        # Display the predicted box
        bd[:4] = _logs[z]
        width, height = (bd[3] - bd[1]) * bd[9], (bd[2] - bd[0]) * bd[8]
        xmin, ymin = bd[1] * bd[9], bd[0] * bd[8]
        cn2 = [ymin + (height // 2), xmin + (width // 2)]
        size2 = [height, width]
        box_pred = sdl.generate_box(_image, cn2, size2, dim3d=False)[0]

        # Make new image
        display = np.zeros([512, 768], np.float32)
        display[0:512, 0:512] = _data['data'][z]
        display[0:256, 512:] = sdl.zoom_2D(box_true, [256, 256])
        display[256:, 512:] = sdl.zoom_2D(box_pred, [256, 256])
        try: display = sdl.adaptive_normalization(display).astype(np.float32)
        except: pass

        # Display
        show_all.append(display)

    return np.asarray(show_all)


def cen_display(_data, _logs):

    """
    Displays the bounding boxes made by a center spot predictor type network
    :return:
    """

    # Trackers
    show_all = []

    # Output box_data = [0ymin, 1xmin, 2ymax, 3xmax, 4cny, 5cnx, 6height, 7width, 8origshapey, 9origshapex]
    # for z in range (FLAGS.epoch_size):
    for z in range(FLAGS.epoch_size):

        # Remake GTBox from saved data
        bd = _data['box_data'][z]
        width, height = (bd[3] - bd[1]) * bd[9], (bd[2] - bd[0]) * bd[8]
        xmin, ymin = bd[1] * bd[9], bd[0] * bd[8]
        cn = [ymin + (height // 2), xmin + (width // 2)]
        size = [height, width]
        _image = sdl.zoom_2D(_data['data'][z], [bd[8], bd[9]])
        box_true = sdl.generate_box(_image, cn, size, dim3d=False)[0]

        # Display the predicted box
        bd[4:6] = _logs[z]
        height, width = 200, 200
        xmin, ymin = (bd[5] * bd[9]) - width/2, (bd[4] * bd[8]) - height/2
        gtbox_recon = {'name': 'rect', 'x': xmin, 'y': ymin, 'width': width, 'height': height}

        # Get parameters for center box
        cn2 = [gtbox_recon['y'] + (gtbox_recon['height'] // 2),
              gtbox_recon['x'] + (gtbox_recon['width'] // 2)]
        size2 = [gtbox_recon['height'], gtbox_recon['width']]
        box_pred = sdl.generate_box(_image, cn2, size2, dim3d=False)[0]

        # Make new image
        display = np.zeros([512, 768], np.float32)
        display[0:512, 0:512] = _data['data'][z]
        display[0:256, 512:] = sdl.zoom_2D(box_true, [256, 256])
        display[256:, 512:] = sdl.zoom_2D(box_pred, [256, 256])
        try:
            display = sdl.adaptive_normalization(display).astype(np.float32)
        except:
            pass

        # Display
        show_all.append(display)

    return np.asarray(show_all)


def main(argv=None):
    test()

if __name__ == '__main__':
    tf.app.run()