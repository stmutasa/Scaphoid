"""
To train the Localization model
"""

import os
import time

import LocModel as network
import numpy as np
import tensorflow as tf
import SODLoader as SDL

sdl= SDL.SODLoader('/home/stmutasa/Code/Datasets/Scaphoid/')

# Define flags
FLAGS = tf.app.flags.FLAGS

# Define some of the data variables
tf.app.flags.DEFINE_string('data_dir', 'data/train/', """Path to the data directory.""")
tf.app.flags.DEFINE_string('training_dir', 'training/', """Path to the training directory.""")
tf.app.flags.DEFINE_integer('box_dims', 1024, """dimensions to save files""")
tf.app.flags.DEFINE_integer('network_dims', 512, """dimensions of the network input""")
tf.app.flags.DEFINE_integer('repeats', 10, """epochs to repeat before reloading""")
tf.app.flags.DEFINE_string('net_type', 'CEN', """Network predicting CEN or BBOX""")

# Define some of the immutable variables
tf.app.flags.DEFINE_integer('num_epochs', 3000, """Number of epochs to run""")
tf.app.flags.DEFINE_integer('epoch_size', 468, """How many examples""")
tf.app.flags.DEFINE_integer('print_interval', 5, """How often to print a summary to console during training""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 25, """How many Epochs to wait before saving a checkpoint""")
tf.app.flags.DEFINE_integer('batch_size', 64, """Number of images to process in a batch.""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ Keep probability""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-3, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")

# Hyperparameters to control the optimizer
tf.app.flags.DEFINE_float('learning_rate',1e-3, """Initial learning rate""")
tf.app.flags.DEFINE_float('beta1', 0.9, """ The beta 1 value for the adam optimizer""")
tf.app.flags.DEFINE_float('beta2', 0.999, """ The beta 1 value for the adam optimizer""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('RunInfo', 'Center/', """Unique file name for this training run""")
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
        data['data'] = tf.reshape(data['data'], [FLAGS.batch_size, FLAGS.network_dims, FLAGS.network_dims])

        # Perform the forward pass:
        if FLAGS.net_type == 'BBOX': logits = network.forward_pass((data['data'], data['img_small']), phase_train=phase_train)
        elif FLAGS.net_type =='CEN': logits = network.forward_pass_center((data['data'], data['img_small']), phase_train=phase_train)
        l2loss = network.sdn.calc_L2_Loss(FLAGS.l2_gamma)

        # Labels
        labels = data['box_data']

        # Calculate loss
        MSE_Loss = network.total_loss(logits, labels, type=FLAGS.net_type)

        # Add the L2 regularization loss
        loss = tf.add(MSE_Loss, l2loss, name='TotalLoss')

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
        saver = tf.train.Saver(var_restore, max_to_keep=10)

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

            # Initialize the handle to the summary writer in our training directory
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir + FLAGS.RunInfo, mon_sess.graph)

            # Initialize the step counter
            timer, batch_count = 0, 0

            # Finalize graph
            mon_sess.graph.finalize()

            # No queues!
            for i in range(max_steps):

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
                        _lbls, _logs, _MSELoss, _l2loss, _loss, _id = mon_sess.run([
                            labels, logits, MSE_Loss, l2loss, loss, data['view']], feed_dict={phase_train: True})

                        # Make losses display in ppt
                        _loss *= 1e3
                        _MSELoss *= 1e3
                        _l2loss *= 1e3

                        # Get timing stats
                        elapsed = timer / print_interval
                        timer = 0

                        # Clip labels
                        if FLAGS.net_type == 'BBOX': _lbls = _lbls[:, :4]
                        elif FLAGS.net_type == 'CEN': _lbls = _lbls[:, 4:6]

                        # use numpy to print only the first sig fig
                        np.set_printoptions(precision=3, suppress=True, linewidth=150)

                        # Calc epoch
                        Epoch = int((i * FLAGS.batch_size) / FLAGS.epoch_size)

                        # Print the data
                        print('-' * 70)
                        print('\nEpoch %d, L2 Loss: = %.3f (%.1f eg/s), Total Loss: %.3f MSE: %.4f'
                              % (Epoch, _l2loss, FLAGS.batch_size / elapsed, _loss, _MSELoss))

                        print('*** MSE = %.4f,  Labels/Logits: ***' % _MSELoss)
                        for z in range(10):
                            this_MAE = np.mean(np.absolute(_logs[z] - _lbls[z]))

                            if FLAGS.net_type == 'BBOX':
                                print('%s -- %.3f/%.3f, %.3f/%.3f, %.3f/%.3f, %.3f/%.3f for an MAE of %.2f%%'
                                      % (_id[z], _lbls[z, 0], _logs[z, 0], _lbls[z, 1], _logs[z, 1], _lbls[z, 2],
                                         _logs[z, 2], _lbls[z, 3], _logs[z, 3], this_MAE * 100))

                            if FLAGS.net_type == 'CEN':
                                print('%s -- %.3f/%.3f, %.3f/%.3f, for an MAE of %.2f%%'
                                      % (_id[z], _lbls[z, 0], _logs[z, 0], _lbls[z, 1], _logs[z, 1], this_MAE * 100))

                        # Run a session to retrieve our summaries
                        summary = mon_sess.run(all_summaries, feed_dict={phase_train: True})

                        # Add the summaries to the protobuf for Tensorboard
                        summary_writer.add_summary(summary, i)

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
    if tf.gfile.Exists(FLAGS.train_dir + FLAGS.RunInfo):
        tf.gfile.DeleteRecursively(FLAGS.train_dir + FLAGS.RunInfo)
    tf.gfile.MakeDirs(FLAGS.train_dir + FLAGS.RunInfo)
    train()


if __name__ == '__main__':
    tf.app.run()