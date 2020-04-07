"""
To train the Localization model
"""

import os
import time

import LocModel as network
import numpy as np
import tensorflow as tf
import SODLoader as SDL
from Input import sdd as sdd

# Define the data directory to use
home_dir = '/home/stmutasa/Code/Datasets/Scaphoid/'
tfrecords_dir = home_dir + 'tfrecords/train/'

sdl= SDL.SODLoader('/home/stmutasa/Code/Datasets/Scaphoid/')


# Define flags
FLAGS = tf.app.flags.FLAGS

# Define some of the data variables
tf.app.flags.DEFINE_string('data_dir', tfrecords_dir, """Path to the data directory.""")
tf.app.flags.DEFINE_string('training_dir', 'training/', """Path to the training directory.""")
tf.app.flags.DEFINE_integer('box_dims', 64, """dimensions to save files""")
tf.app.flags.DEFINE_integer('network_dims', 64, """dimensions of the network input""")
tf.app.flags.DEFINE_integer('repeats', 20, """epochs to repeat before reloading""")
tf.app.flags.DEFINE_string('net_type', 'RPNC', """Network predicting CEN, BBOX or RPN""")

# Define some of the immutable variables
tf.app.flags.DEFINE_integer('num_epochs', 202, """Number of epochs to run""")
tf.app.flags.DEFINE_integer('epoch_size', 19237152, """10 Million in train set""")
tf.app.flags.DEFINE_integer('print_interval', 1, """How often to print a summary to console during training""")
tf.app.flags.DEFINE_float('checkpoint_interval', 5, """How many Epochs to wait before saving a checkpoint""")
tf.app.flags.DEFINE_integer('batch_size', 1280, """Number of images to process in a batch.""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.75, """ Keep probability""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-3, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")

# Hyperparameters to control the optimizer
tf.app.flags.DEFINE_float('learning_rate',1e-3, """Initial learning rate""")
tf.app.flags.DEFINE_float('beta1', 0.9, """ The beta 1 value for the adam optimizer""")
tf.app.flags.DEFINE_float('beta2', 0.999, """ The beta 1 value for the adam optimizer""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('RunInfo', 'RPN_FL6/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 1, """Which GPU to use""")


def train():

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('/gpu:' + str(FLAGS.GPU)):

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Load the images and labels.
        iterator = network.inputs(training=True, skip=True)
        data = iterator.get_next()

        # Define input shape
        data['data'] = tf.reshape(data['data'], [FLAGS.batch_size, FLAGS.network_dims, FLAGS.network_dims, 1])

        # Perform the forward pass:
        all_logits = network.forward_pass_RPN(data['data'], phase_train=phase_train)
        l2loss = network.sdn.calc_L2_Loss(FLAGS.l2_gamma)

        # Labels and logits
        labels = data['box_data']
        logits = tf.nn.softmax(all_logits[0])

        # Calculate loss
        combined_loss, class_loss, loc_loss = network.total_loss(all_logits, labels)

        # Add the L2 regularization loss
        loss = tf.add(combined_loss, l2loss, name='TotalLoss')

        # Update the moving average batch norm ops
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Retreive the training operation with the applied gradients
        with tf.control_dependencies(extra_update_ops): train_op = network.backward_pass(loss)

        # -------------------  Housekeeping functions  ----------------------

        # Merge the summaries
        all_summaries = tf.summary.merge_all()

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=30)

        # -------------------  Session Initializer  ----------------------

        # Set the intervals
        max_steps = int((FLAGS.epoch_size / FLAGS.batch_size) * FLAGS.num_epochs) + 5
        print_interval = int((FLAGS.epoch_size / FLAGS.batch_size) * FLAGS.print_interval) + 1
        checkpoint_interval = int((FLAGS.epoch_size / FLAGS.batch_size) * FLAGS.checkpoint_interval) + 1
        print('Max Steps: %s, Print Interval: %s, Checkpoint: %s' % (max_steps, print_interval, checkpoint_interval))

        # Print Run info
        print("*** Training Run %s on GPU %s ****" % (FLAGS.RunInfo, FLAGS.GPU))

        # Allow memory placement growth
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as mon_sess:

            # Initialize the variables and iterator
            mon_sess.run(var_init)
            mon_sess.run(iterator.initializer)

            # TODO: Restore the model
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + FLAGS.RunInfo)
            saver.restore(mon_sess, ckpt.model_checkpoint_path)
            print ('Model restored: ', ckpt.model_checkpoint_path)
            start_step = 751451

            # Initialize the handle to the summary writer in our training directory
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir + FLAGS.RunInfo, mon_sess.graph)

            # Initialize the step counter
            timer, batch_count = 0, 0

            # Finalize graph
            mon_sess.graph.finalize()

            # No queues!
            # for i in range(max_steps):
            for i in range(start_step, max_steps, 1):

                # Run and time an iteration
                start = time.time()
                try:
                    mon_sess.run(train_op, feed_dict={phase_train: True})
                    batch_count += FLAGS.batch_size
                except tf.errors.OutOfRangeError:
                    print('*' * 10, '\n%s examples run, re-initializing iterator\n' % batch_count)
                    batch_count = 0
                    mon_sess.run(iterator.initializer)
                timer += (time.time() - start)

                # Calculate current epoch
                Epoch = int((i * FLAGS.batch_size) / FLAGS.epoch_size)

                try:

                    # Console and Tensorboard print interval
                    if i % print_interval == 0:

                        # Load some metrics
                        _lbls, _logs, _combinedLoss, _clsLoss, _locLoss, _l2loss, _totLoss, _id = mon_sess.run([
                            labels, logits, combined_loss, class_loss, loc_loss, l2loss, loss, data['view']],
                            feed_dict={phase_train: True})

                        # Make losses display in ppt
                        _totLoss *= 1e3
                        _combinedLoss *= 1e3
                        _l2loss *= 1e3 * FLAGS.l2_gamma
                        _clsLoss *= 1e3
                        _locLoss *= 1e3

                        # Positive count
                        pct = np.sum(_lbls[:, 19])

                        # Retreive the indices
                        class_preds = np.argmax(_logs, axis=1)

                        # Get metrics for this batch:
                        TP = np.sum(np.logical_and(class_preds == 1, _lbls[:, 19] == 1))
                        TN = np.sum(np.logical_and(class_preds == 0, _lbls[:, 19] == 0))
                        FP = np.sum(np.logical_and(class_preds == 1, _lbls[:, 19] == 0))
                        FN = np.sum(np.logical_and(class_preds == 0, _lbls[:, 19] == 1))

                        # Calculate stats
                        SN, SP = TP / (TP + FN), TN / (TN + FP)
                        PPV, NPV = TP / (TP + FP), TN / (TN + FN)
                        Acc = 100 * (TP + TN) / (TN + TP + FN + FP)

                        # Get timing stats
                        elapsed = timer / print_interval
                        timer = 0

                        # Clip labels
                        _lblsCls = _lbls[:, 19]

                        # use numpy to print only the first sig fig
                        np.set_printoptions(precision=3, suppress=True, linewidth=150)

                        # Calc epoch
                        Epoch = int((i * FLAGS.batch_size) / FLAGS.epoch_size)

                        # Saved the data to [0ymin, 1xmin, 2ymax, 3xmax, cny, cnx, 6height, 7width, 8origshapey, 9origshapex,
                        #    10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, 18IOU, 19obj_class, 20#_class]

                        # Print the data
                        print('-' * 70)
                        print('\nEpoch %d, Losses: L2:%.3f, Comb:%.3f, Class:%.3f, Loc:%.3f,  (%.1f eg/s), Total Loss: %.3f '
                              % (Epoch, _l2loss, _combinedLoss, _clsLoss, _locLoss, FLAGS.batch_size / elapsed, _totLoss))

                        # Print the stats
                        print('*** Sn:%.3f, Sp:%.3f, PPv:%.3f, NPv:%.3f ***' % (SN, SP, PPV, NPV))
                        print('*** Acc:%.2f TP:%s, TN:%s, FP:%s, FN:%s ***' % (Acc, TP, TN, FP, FN))

                        # Print examples
                        print('*** Pos in Batch %s of %s,  Labels/Logits: ***' % (pct, FLAGS.batch_size))
                        for z in range(0, 1000, 100):
                            print('%s -- Class Label: %s, Pred %s %s' % (
                            _id[z], _lblsCls[z], np.argmax(_logs[z]), _logs[z]))

                        # Run a session to retrieve our summaries
                        summary = mon_sess.run(all_summaries, feed_dict={phase_train: True})

                        # Add the summaries to the protobuf for Tensorboard
                        summary_writer.add_summary(summary, i)

                        # Garbage cleanup
                        del _lbls, _logs, _combinedLoss, _clsLoss, _locLoss, _l2loss, _totLoss, _id

                    if i % checkpoint_interval == 0:

                        print('-' * 70, '\nSaving... GPU: %s, File:%s' % (FLAGS.GPU, FLAGS.RunInfo[:-1]))

                        # Define the filename
                        file = ('Epoch_%s' % Epoch)

                        # Define the checkpoint file:
                        checkpoint_file = os.path.join(FLAGS.train_dir + FLAGS.RunInfo, file)

                        # Save the checkpoint
                        saver.save(mon_sess, checkpoint_file)

                        # Sleep a little to let testing catch up
                        time.sleep(5)

                except tf.errors.OutOfRangeError:
                    print('*' * 10, time.time(), '\nOut of Range error: re-initializing iterator')
                    batch_count = 0
                    mon_sess.run(iterator.initializer)


def main(argv=None):
    # if tf.gfile.Exists(FLAGS.train_dir + FLAGS.RunInfo):
    #     tf.gfile.DeleteRecursively(FLAGS.train_dir + FLAGS.RunInfo)
    # tf.gfile.MakeDirs(FLAGS.train_dir + FLAGS.RunInfo)
    train()


if __name__ == '__main__':
    tf.app.run()