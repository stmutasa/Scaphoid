""" Evaluating the localization network on a CPU or GPU """


import LocModel as network
import tensorflow as tf
import SODLoader as SDL
import SOD_Display as SDD
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
tf.app.flags.DEFINE_integer('epoch_size', 2449062, """How many examples""")
tf.app.flags.DEFINE_integer('batch_size', 5301, """Number of images to process in a batch.""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.75, """ Keep probability""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory where to retrieve checkpoint files""")
tf.app.flags.DEFINE_string('net_type', 'RPNC', """Network predicting CEN or BBOX""")
tf.app.flags.DEFINE_string('RunInfo', 'RPN_FL2/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 0, """Which GPU to use""")

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
        data['data'] = tf.reshape(data['data'], [FLAGS.batch_size, FLAGS.network_dims, FLAGS.network_dims])

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

            # Initialize the variables
            mon_sess.run([var_init, iterator.initializer])

            # Restore the model
            saver.restore(mon_sess, ckpt.model_checkpoint_path)
            Epoch = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1]

            # Print run info
            print("*** Epoch %s Generating Run %s on GPU %s ****" % (Epoch, FLAGS.RunInfo, FLAGS.GPU))

            # Initialize the step counter
            step, made = 0, False

            # Set the max step count
            max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)

            # Dict to save
            save_dict, index, PROPS = {}, 0, []

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
                        save_dict[index] = {'data': _data['data'][idx],
                                        'box_data': _data['box_data'][idx],
                                        'group': _data['group'][idx].decode('utf-8'),
                                        'view': _data['view'][idx].decode('utf-8'),
                                        'accno': _data['accno'][idx].decode('utf-8'),
                                        'obj_prob':_softmax[idx, 1]}
                        text = ('%.2f-%s' % (_softmax[idx, 1], save_dict[index]['accno']))
                        PROPS.append(sdd.return_image_text_overlay(text, _data['data'][idx], scale=0.25))
                        index +=1

                    # Increment step
                    del _softmax, _data, class_preds
                    step += 1

            except tf.errors.OutOfRangeError:
                print('Done with Training - Epoch limit reached')

            finally:

                # Save
                print ('%s Objects generated from %s proposals' %(len(save_dict), FLAGS.epoch_size))
                sdd.display_volume(PROPS, True)

                # Shut down the session
                del PROPS
                mon_sess.close()


def main(argv=None):
    test()

if __name__ == '__main__':
    tf.app.run()