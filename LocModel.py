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


def forward_pass(images, phase_train):

    """
    Train a 2 dimensional network. Default input size 512x512
    :param images: tuple (full size images, scaled down by 8 aka 64)
    :param phase_train: True if this is the training phase
    :return: L2 Loss and Logits
    """

    # Initial kernel size
    K = 4
    scaled_images = images[1]
    images = tf.expand_dims(images[0], -1)

    # Channel wise layers. Inputs = batchx512x512
    conv = sdn.convolution('Conv1', images, 3, K, phase_train=phase_train) # 256
    conv = sdn.convolution('Conv2', conv, 3, K * 2, phase_train=phase_train) # 128
    conv = sdn.convolution('Conv2b', conv, 3, K * 2, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv3', conv, 3, K * 4, phase_train=phase_train) # 64
    # Add scaled image
    conv = tf.concat([conv, scaled_images], -1)
    conv = sdn.inception_layer('Conv3a', conv, K * 4, S=1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv3b', conv, 3, K * 4, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv4', conv, 3, K * 8, phase_train=phase_train) # 32
    conv = sdn.residual_layer('Conv4b', conv, 3, K * 8, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv5', conv, K * 16, S=2, phase_train=phase_train) # 16
    conv = sdn.inception_layer('Conv6', conv, K * 32, S=2, phase_train=phase_train)  # 8

    # Split top left and bottom right here
    conv_TL = sdn.residual_layer('Conv_TopLeft', conv, 3, K * 64, S=2, phase_train=phase_train)
    conv_BR = sdn.residual_layer('Conv_BotRight', conv, 3, K * 64, S=2, phase_train=phase_train)

    # Linear layers for top left
    linearTL = sdn.fc7_layer('FC7TL', conv_TL, 8, True, phase_train, FLAGS.dropout_factor, override=3, BN=True)
    linearTL = sdn.linear_layer('LinearTL', linearTL, 4, True, phase_train, FLAGS.dropout_factor, BN=True)
    LogitsTL = sdn.linear_layer('SoftmaxTL', linearTL, 2, relu=False, add_bias=False, BN=False)
    # Linear layers for bottom right
    linearBR = sdn.fc7_layer('FC7BR', conv_BR, 8, True, phase_train, FLAGS.dropout_factor, override=3, BN=True)
    linearBR = sdn.linear_layer('LinearBR', linearBR, 4, True, phase_train, FLAGS.dropout_factor, BN=True)
    LogitsBR = sdn.linear_layer('SoftmaxBR', linearBR, 2, relu=False, add_bias=False, BN=False)

    # Add logits along batchxnum axis 1
    Logits = tf.concat([LogitsTL, LogitsBR], 1)

    return Logits


def forward_pass_center(images, phase_train):

    """
    Train a 2 dimensional network. Default input size 512x512
    :param images: tuple (full size images, scaled down by 8 aka 64)
    :param phase_train: True if this is the training phase
    :return: L2 Loss and Logits
    """

    # Initial kernel size
    K = 4
    scaled_images = images[1]
    images = tf.expand_dims(images[0], -1)

    # Channel wise layers. Inputs = batchx512x512
    conv = sdn.convolution('Conv1', images, 3, K, phase_train=phase_train) # 256
    conv = sdn.convolution('Conv2', conv, 3, K * 2, phase_train=phase_train) # 128
    conv = sdn.convolution('Conv2b', conv, 3, K * 2, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv3', conv, 3, K * 4, phase_train=phase_train) # 64
    # Add scaled image
    conv = tf.concat([conv, scaled_images], -1)
    conv = sdn.inception_layer('Conv3a', conv, K * 4, S=1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv3b', conv, 3, K * 4, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv4', conv, 3, K * 8, phase_train=phase_train) # 32
    conv = sdn.residual_layer('Conv4b', conv, 3, K * 8, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv5', conv, K * 16, S=2, phase_train=phase_train) # 16
    conv = sdn.inception_layer('Conv6', conv, K * 32, S=2, phase_train=phase_train)  # 8
    conv = sdn.residual_layer('Conv7', conv, 3, K * 64, S=2, phase_train=phase_train)

    # Linear layers for center
    linear = sdn.fc7_layer('FC7', conv, 8, True, phase_train, FLAGS.dropout_factor, override=3, BN=True)
    linear = sdn.linear_layer('Linear', linear, 4, True, phase_train, FLAGS.dropout_factor, BN=True)
    Logits = sdn.linear_layer('Softmax', linear, 2, relu=False, add_bias=False, BN=False)

    return Logits


def total_loss(logits, labels, type='BBOX'):

    """
    Add loss to the trainable variables and a summary
    box_data = [0ymin, 1xmin, 2ymax, 3xmax, 4cny, 5cnx, 6height, 7width, 8origshapey, 9origshapex]
    """

    # Squish
    labels = tf.squeeze(labels)
    logits = tf.squeeze(logits)

    # Calculate loss for bounding box
    if type=='BBOX':
        loss = 0
        for var in range(4):
            loss += tf.reduce_mean(tf.square(logits[:, var] - labels[:, var]))

    # Calculate loss for center
    elif type=='CEN':
        lossy = tf.reduce_mean(tf.square(logits[:, 0] - labels[:, 4]))
        lossx = tf.reduce_mean(tf.square(logits[:, 1] - labels[:, 5]))
        loss = lossy + lossx

    # Output the summary of the MSE and MAE
    tf.summary.scalar('MAE_Loss', loss)

    # Add these losses to the collection
    tf.add_to_collection('losses', loss)

    return loss


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
    dk_steps = int((FLAGS.epoch_size / FLAGS.batch_size) * 125)
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