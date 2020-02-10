# Defines and builds the localization network
#    Computes input images and labels using inputs()
#    Computes inference on the models (forward pass) using inference()
#    Computes the total loss using loss()
#    Performs the backprop using train()

_author_ = 'simi'

import tensorflow as tf
import LocInput as Input
import SODNetwork as SDN

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Retreive helper function object
sdn = SDN.SODMatrix()
sdloss = SDN.SODLoss(2)


def forward_pass_RPN(images, phase_train):

    """
    Train a 2 dimensional network. Default input size 64x64
    :param images: tuple (full size images, scaled down by 8 aka 64)
    :param phase_train: True if this is the training phase
    :return: L2 Loss and Logits
    """

    # Initial kernel size
    K = 8
    images = tf.expand_dims(images[0], -1)

    # Channel wise layers. Inputs = batchx64x64x64
    conv = sdn.residual_layer('Conv1', images, 3, K, S=1, phase_train=phase_train) # 64
    conv = sdn.inception_layer('Conv1ds', conv, K * 2, S=2, phase_train=phase_train) # 32

    conv = sdn.residual_layer('Conv2a', conv, 3, K * 2, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv2b', conv, 3, K * 2, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv2ds', conv, K * 4, S=2, phase_train=phase_train)  # 32

    conv = sdn.residual_layer('Conv3a', conv, 3, K * 4, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv3b', conv, 3, K * 4, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv3c', conv, 3, K * 4, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv3ds', conv, K * 8, S=2, phase_train=phase_train)  # 16

    conv = sdn.residual_layer('Conv4a', conv, 3, K * 8, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv4b', conv, K * 8, S=1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv4c', conv, 3, K * 8, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv4ds', conv, K * 16, S=2, phase_train=phase_train)  # 8

    conv = sdn.residual_layer('Conv5a', conv, 3, K * 16, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv5b', conv, K * 16, S=1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv5c', conv, K * 16, S=1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv5d', conv, K * 32, S=2, phase_train=phase_train)  # 4

    conv = sdn.residual_layer('Conv6a', conv, 3, K * 32, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv6b', conv, K * 32, S=1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv6c', conv, 3, K * 32, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv6d', conv, 3, K * 32, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv6e', conv, 3, K * 32, 1, phase_train=phase_train)

    # TODO: RPN has class logits and BBox logits, need another branch

    # Linear layers for center
    linear = sdn.fc7_layer('FC7', conv, 8, True, phase_train, FLAGS.dropout_factor, override=3, BN=True)
    linear = sdn.linear_layer('Linear', linear, 4, True, phase_train, FLAGS.dropout_factor, BN=True)
    Logits = sdn.linear_layer('Softmax', linear, 2, relu=False, add_bias=False, BN=False)

    return Logits


def total_loss(logits, labels):

    """
    Add loss to the trainable variables and a summary
    """

    # Saved the data to [0ymin, 1xmin, 2ymax, 3xmax, cny, cnx, 6height, 7width, 8origshapey, 9origshapex,
    #    10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, 18IOU, 19obj_class, 20#_class]


    # Squish
    labels = tf.squeeze(labels)
    logits = tf.squeeze(logits)

    # Make an object mask and multiply this mask by scaling factor then add back to labels. Add 1 to prevent 0 loss
    object_mask = tf.cast(labels[19] >= 0.1, tf.float32)
    object_mask = tf.add(tf.multiply(object_mask, FLAGS.loss_factor), 1)

    # Calculate loss for bounding box only if it's on an object with ROI > 0.05
    # TODO: Apply object mask
    # if FLAGS.loss_factor != 1.0: loss = tf.multiply(loss, tf.squeeze(lesion_mask))
    L1_loss, L2_loss = 0, 0
    for var in range(4):
        L2_loss += tf.reduce_mean(tf.square(logits[:, var] - labels[:, var]))
        L1_loss += tf.reduce_mean(tf.abs(logits[:, var] - labels[:, var]))
    loc_loss = (L1_loss + L2_loss) / 2


    # Change labels to one hot
    labels = tf.one_hot(tf.cast(labels[19], tf.uint8), depth=FLAGS.num_classes, dtype=tf.uint8)

    # Calculate  loss
    class_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.squeeze(labels), logits=logits)

    # Reduce to scalar
    class_loss = tf.reduce_mean(class_loss)

    # Add losses
    total_loss = 10*class_loss + loc_loss

    # Output the summary of the MSE and MAE
    tf.summary.scalar('Class_Loss', class_loss)
    tf.summary.scalar('Loc_Loss', loc_loss)
    tf.summary.scalar('Tot_Loss', total_loss)

    # Add these losses to the collection
    tf.add_to_collection('losses', class_loss)
    tf.add_to_collection('losses', loc_loss)
    tf.add_to_collection('losses', total_loss)

    return total_loss


def backward_pass(total_loss):

    """
    This function performs our backward pass and updates our gradients
    """

    # Get the tensor that keeps track of step in this graph or create one if not there
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Print summary of total loss
    tf.summary.scalar('Total_Loss', total_loss)

    # Decay the learning rate
    #dk_steps = int((FLAGS.epoch_size / FLAGS.batch_size) * (FLAGS.num_epochs/4))
    dk_steps = int((FLAGS.epoch_size / FLAGS.batch_size) * 15)
    lr_decayed = tf.train.cosine_decay_restarts(FLAGS.learning_rate, global_step, dk_steps)

    # Compute the gradients. NAdam optimizer came in tensorflow 1.2
    opt = tf.contrib.opt.NadamOptimizer(learning_rate=lr_decayed, beta1=FLAGS.beta1,
                                        beta2=FLAGS.beta2, epsilon=0.1)

    # Compute the gradients
    gradients = opt.compute_gradients(total_loss)

    # Apply the gradients
    train_op = opt.apply_gradients(gradients, global_step, name='train')

    # Add histograms for the trainable variables. i.e. the collection of variables created with Trainable=True
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Maintain average weights to smooth out training
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay, global_step)

    # Applies the average to the variables in the trainable ops collection
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([train_op, variable_averages_op]):  # Wait until we apply the gradients
        dummy_op = tf.no_op(name='train')  # Does nothing. placeholder to control the execution of the graph

    return dummy_op


def inputs(training=True, skip=False):

    """
    Loads the inputs
    :param filenames: Filenames placeholder
    :param training: if training phase
    :param skip: Skip generating tfrecords if already done
    :return:
    """

    if not skip:  Input.pre_proc_localizations(FLAGS.box_dims)

    else: print('-------------------------Previously saved records found! Loading...')

    return Input.load_protobuf(training)