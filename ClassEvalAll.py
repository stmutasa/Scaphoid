""" Evaluating the localization network on a CPU or GPU """

import ClassModel as network
import tensorflow as tf
import SODTester as SDT
import SODLoader as SDL
import SOD_Display as SDD
import numpy as np
import os

# Define the data directory to use
home_dir = '/home/stmutasa/Code/Datasets/Scaphoid/'
tfrecords_dir = home_dir + 'tfrecords/test/'

sdl= SDL.SODLoader('/home/stmutasa/Code/Datasets/Scaphoid/')
sdd = SDD.SOD_Display()

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the data variables
tf.app.flags.DEFINE_string('data_dir', 'data/test/', """Path to the data directory.""")
tf.app.flags.DEFINE_string('training_dir', 'training/', """Path to the training directory.""")
tf.app.flags.DEFINE_integer('box_dims', 64, """dimensions to save files""")
tf.app.flags.DEFINE_integer('network_dims', 64, """dimensions of the network input""")
tf.app.flags.DEFINE_integer('epoch_size', 1024, """How many examples""")
tf.app.flags.DEFINE_integer('batch_size', 1024, """Number of images to process in a batch.""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.75, """ Keep probability""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory where to retrieve checkpoint files""")
tf.app.flags.DEFINE_string('net_type', 'RPNC', """Network predicting CEN or BBOX""")
tf.app.flags.DEFINE_string('RunInfo', 'Class2/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 1, """Which GPU to use""")

# Define a custom training class
def test():

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('/gpu:' + str(FLAGS.GPU)):

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Load the images and labels.
        iterator = network.inputs(training=False, skip=True)
        data = iterator.get_next()

        # Define input shape
        data['data'] = tf.reshape(data['data'], [FLAGS.batch_size, FLAGS.network_dims, FLAGS.network_dims, 1])

        # Perform the forward pass:
        logits = network.forward_pass_RPN(data['data'], phase_train=phase_train)

        # Labels and logits
        labels = data['box_data']
        softmax = tf.nn.softmax(logits)

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=2)

        # Trackers for best performers
        best_score, best_epoch = 0.1, 0

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
                        __lbls, __smx, __accno = mon_sess.run([labels, softmax, data['accno']], feed_dict={phase_train: False})

                        # Combine metrics
                        if step == 0:
                            _lbls = np.copy(__lbls)
                            _smx = np.copy(__smx)
                            _accnos = np.copy(__accno)
                        else:
                            _lbls = np.concatenate([_lbls, __lbls])
                            _smx = np.concatenate([_smx, __smx])
                            _accnos = np.concatenate([_accnos, __accno])

                        # Increment step
                        del __lbls, __smx, __accno
                        step += 1

                except tf.errors.OutOfRangeError:
                    print('Done with Training - Epoch limit reached')

                finally:

                    # Combine metrics
                    _data, _labels, _softmax = sdt.combine_predictions(_lbls, _smx, _accnos, FLAGS.batch_size)

                    # Retreive the scores
                    sdt.calculate_metrics(_softmax, _labels, 1, step)
                    sdt.retreive_metrics_classification(Epoch, True)
                    print('------ Current Best AUC: %.4f (Epoch: %s) --------' % (best_MAE, best_epoch))

                    # Lets save runs that perform well
                    if sdt.AUC >= best_MAE:

                        # Save the checkpoint
                        print(" ---------------- SAVING THIS ONE %s", ckpt.model_checkpoint_path)

                        # Define the filenames
                        checkpoint_file = os.path.join('testing/' + FLAGS.RunInfo,
                                                       ('Epoch_%s_AUC_%0.3f' % (Epoch, sdt.AUC)))
                        csv_file = os.path.join('testing/' + FLAGS.RunInfo,
                                                ('%s_E_%s_AUC_%0.2f.csv' % (FLAGS.RunInfo[:-1], Epoch, sdt.AUC)))

                        # Save the checkpoint
                        saver.save(mon_sess, checkpoint_file)
                        sdl.save_Dict_CSV(_data, csv_file)

                        # Save a new best MAE
                        best_MAE = sdt.AUC
                        best_epoch = Epoch

                    # Shut down the session
                    del _data, _labels, _softmax, _lbls, _smx, _accnos
                    mon_sess.close()


def main(argv=None):
    test()

if __name__ == '__main__':
    tf.app.run()