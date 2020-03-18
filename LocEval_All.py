""" Evaluating the localization network on a CPU or GPU """

import os
import time

import LocModel as network
import tensorflow as tf
import SODTester as SDT
import SODLoader as SDL
import SOD_Display as SDD
import glob
import numpy as np

# Define the data directory to use
home_dir = '/home/stmutasa/Code/Datasets/Scaphoid/'
tfrecords_dir = home_dir + 'tfrecords/test/'

sdl= SDL.SODLoader('/home/stmutasa/Code/Datasets/Scaphoid/')
sdd = SDD.SOD_Display()

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the data variables
tf.app.flags.DEFINE_string('data_dir', tfrecords_dir, """Path to the data directory.""")
tf.app.flags.DEFINE_string('training_dir', 'training/', """Path to the training directory.""")
tf.app.flags.DEFINE_integer('box_dims', 64, """dimensions to save files""")
tf.app.flags.DEFINE_integer('network_dims', 64, """dimensions of the network input""")
tf.app.flags.DEFINE_integer('epoch_size', 2255034, """How many examples""")
tf.app.flags.DEFINE_integer('batch_size', 7468, """Number of images to process in a batch.""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.75, """ Keep probability""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory where to retrieve checkpoint files""")
tf.app.flags.DEFINE_string('net_type', 'RPNC', """Network predicting CEN or BBOX""")
tf.app.flags.DEFINE_string('RunInfo', 'RPN_FL6/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 1, """Which GPU to use""")

# Define a custom training class
def test():


    # Makes this the default graph where all ops will be added
    #with tf.Graph().as_default(), tf.device('/cpu:0'):
    with tf.Graph().as_default(), tf.device('/gpu:' + str(FLAGS.GPU)):

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Load the images and labels.
        iterator = network.inputs(training=False, skip=True)
        data = iterator.get_next()

        # Define input shape
        data['data'] = tf.reshape(data['data'], [FLAGS.batch_size, FLAGS.network_dims, FLAGS.network_dims, 1])

        # Perform the forward pass:
        all_logits = network.forward_pass_RPN(data['data'], phase_train=phase_train)

        # Labels and logits
        labels = data['box_data']
        logits = tf.nn.softmax(all_logits[0])

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=3)

        # Trackers for best performers
        best_score, best_epoch = 0.1, 0

        # Tester instance
        sdt = SDT.SODTester(False, True)

        # Allow memory placement growth
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        # Retreive the checkpoint
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + FLAGS.RunInfo)
        for checkpoint in ckpt.all_model_checkpoint_paths:

            with tf.Session(config=config) as mon_sess:

                # Initialize the variables
                mon_sess.run([var_init, iterator.initializer])

                saver.restore(mon_sess, checkpoint)
                Epoch = checkpoint.split('/')[-1].split('Epoch')[-1]
                print("*** Testing Checkpoint %s Run %s on GPU %s ****" % (checkpoint, FLAGS.RunInfo, FLAGS.GPU))

                # Initialize the step counter
                step, made = 0, False

                # Init stats
                TP, TN, FP, FN = 0, 0, 0, 0

                # Set the max step count
                max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)

                # Tester instance
                sdt = SDT.SODTester(True, False)

                try:
                    while step < max_steps:

                        # Load some metrics for testing
                        __lbls, __logs, _accno = mon_sess.run([labels, logits, data['accno']], feed_dict={phase_train: False})

                        # Only keep positive
                        positives_here = np.sum(__lbls[:, 19])
                        if positives_here == 0:
                            del __lbls, __logs, _accno
                            step += 1
                            continue

                        # Retreive the indices
                        class_preds = np.argmax(__logs, axis=1)
                        tp_indices = np.where(np.equal(__lbls[:, 19] + class_preds, 2))[0]

                        # Get metrics for this batch:
                        TP += np.sum(np.logical_and(class_preds == 1, __lbls[:, 19] == 1))
                        TN += np.sum(np.logical_and(class_preds == 0, __lbls[:, 19] == 0))
                        FP += np.sum(np.logical_and(class_preds == 1, __lbls[:, 19] == 0))
                        FN += np.sum(np.logical_and(class_preds == 0, __lbls[:, 19] == 1))

                        if not made:
                            Unique = np.take(_accno, tp_indices, axis=0)
                            all_counts = _accno
                            made = True
                        else:
                            Unique = np.concatenate([Unique, np.take(_accno, tp_indices, axis=0)])
                            all_counts = np.concatenate([all_counts, _accno])

                        # Increment step
                        del __lbls, __logs, _accno, class_preds, tp_indices
                        step += 1

                except tf.errors.OutOfRangeError:
                    print('Done with Training - Epoch limit reached')

                finally:

                    # Calculate final stats
                    SN, SP = TP / (TP + FN), TN / (TN + FP)
                    PPV, NPV = TP / (TP + FP), TN / (TN + FN)
                    Acc = 100 * (TP + TN) / (TN+TP+FN+FP)
                    Unique, Counts = np.unique(Unique, return_counts=True)
                    Unique_all = np.unique(Unique)
                    pct_Unq = 100 * Unique.shape[0] / (Unique_all.shape[0] + 1e-8)
                    try: ucnts = np.argmin(Counts)
                    except: ucnts = 'Error'

                    # Best score
                    score = (SN + SP) /2
                    if score >= best_score:
                        best_score = score
                        best_epoch = Epoch

                    # Print
                    print('\n*** Sn:%.3f, Sp:%.3f, PPv:%.3f, NPv:%.3f ***' % (SN, SP, PPV, NPV))
                    print('*** Acc:%.2f TP:%s, TN:%s, FP:%s, FN:%s ***' % (Acc, TP, TN, FP, FN))
                    print("Epoch:%s (Best %s-%.3f), Accnos with True Positives: %s of %s (%s%% - As little as %s)\n" %
                          (Epoch, best_epoch, best_score, Unique.shape[0], Unique_all.shape[0], pct_Unq, ucnts))
                    sdt.MAE = SN

                    # Shut down the session
                    del Unique_all, Unique
                    mon_sess.close()



def main(argv=None):  # pylint: disable=unused-argument
    # time.sleep(3600*9)
    test()

if __name__ == '__main__':
    tf.app.run()