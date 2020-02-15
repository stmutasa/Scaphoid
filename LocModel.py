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
    images = tf.expand_dims(images, -1)
    images = tf.cast(images, tf.float32)

    # Channel wise layers. Inputs = batchx64x64x64
    conv = sdn.residual_layer('Conv1', images, 3, K, S=1, phase_train=phase_train) # 64
    conv = sdn.inception_layer('Conv1ds', conv, K * 2, S=2, phase_train=phase_train) # 32

    conv = sdn.residual_layer('Conv2a', conv, 3, K * 2, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv2b', conv, 3, K * 2, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv2ds', conv, K * 4, S=2, phase_train=phase_train)  # 16

    conv = sdn.residual_layer('Conv3a', conv, 3, K * 4, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv3b', conv, 3, K * 4, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv3c', conv, 3, K * 4, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv3ds', conv, K * 8, S=2, phase_train=phase_train)  # 8

    conv = sdn.residual_layer('Conv4a', conv, 3, K * 8, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv4b', conv, K * 8, S=1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv4c', conv, 3, K * 8, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv4d', conv, 3, K * 8, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv4es', conv, K * 16, S=2, phase_train=phase_train)  # 4

    conv = sdn.residual_layer('Conv5a', conv, 3, K * 16, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv5b', conv, K * 16, S=1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv5c', conv, 3, K * 16, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv5d', conv, K * 16, S=1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv5e', conv, 3, K * 16, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv5f', conv, 3, K * 16, 1, phase_train=phase_train)

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
    labels = tf.cast(tf.squeeze(labels), tf.float32)
    logits = tf.squeeze(logits)

    # Make an object mask and multiply this mask by scaling factor then add back to labels. Add 1 to prevent 0 loss
    object_mask = tf.cast(labels[:, 19] >= 0.1, tf.float32)
    object_mask = tf.add(tf.multiply(object_mask, 1), 1)

    # Calculate loss for bounding box only if it's on an object with ROI > 0.05
    # TODO: Apply object mask
    loc_loss = 0

    # # if FLAGS.loss_factor != 1.0: loss = tf.multiply(loss, tf.squeeze(lesion_mask))
    # L1_loss, L2_loss = 0, 0
    # for var in range(4):
    #     L2_loss += tf.reduce_mean(tf.square(logits[:, var] - labels[:, var]))
    #     L1_loss += tf.reduce_mean(tf.abs(logits[:, var] - labels[:, var]))
    # loc_loss = (L1_loss + L2_loss) / 2


    # Change labels to one hot
    labels = tf.one_hot(tf.cast(labels[:, 19], tf.uint8), depth=2, dtype=tf.uint8)

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


def _find_deltas(boxes, anchors, scale_factors=None):
    """
    Generate deltas to transform source offsets into destination anchors
    :param boxes: BoxList holding N boxes to be encoded.
    :param anchors: BoxList of anchors.
    :param: scale_factors: scales location targets when using joint training
    :return:
      a tensor representing N anchor-encoded boxes of the format
      [ty, tx, th, tw].
    """

    # Convert anchors to the center coordinate representation.
    ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
    ymia, xmia, ymaa, xmaa = tf.unstack(anchors, axis=1)
    cenx = (xmin + xmax) / 2
    cenxa = cenxa = (xmia + xmaa) / 2
    ceny = (ymin + ymax) / 2
    cenya = (ymia + ymaa) / 2
    w = xmax - xmin
    h = ymax - ymin
    wa = xmaa - xmia
    ha = ymaa - ymia

    # Avoid NaN in division and log below.
    ha += 1e-8
    wa += 1e-8
    h += 1e-8
    w += 1e-8

    # Calculate the normalized translations required
    tx = (cenx - cenxa) / wa
    ty = (ceny - cenya) / ha
    tw = tf.log(w / wa)
    th = tf.log(h / ha)

    # Scales location targets as used in paper for joint training.
    if scale_factors:
        ty *= scale_factors[0]
        tx *= scale_factors[1]
        th *= scale_factors[2]
        tw *= scale_factors[3]

    return tf.transpose(tf.stack([ty, tx, th, tw]))


def _smooth_l1_loss(self, predicted_boxes, gtboxes, object_weights, classes_weights=None):

    """
    TODO: Calculates the smooth L1 losses
    :param predicted_boxes: The filtered anchors
    :param gtboxes:ground truth boxes
    :param object_weights: The mask map indicating whether this is an object or not
    :return:
    """

    diff = predicted_boxes - gtboxes
    absolute_diff = tf.cast(tf.abs(diff), tf.float32)

    if classes_weights is None:

        anchorwise_smooth_l1norm = tf.reduce_sum(tf.where(tf.less(absolute_diff, 1),0.5 * tf.square(absolute_diff), absolute_diff - 0.5),
                                                 axis=1) * object_weights

    else:

        anchorwise_smooth_l1norm = tf.reduce_sum(tf.where(tf.less(absolute_diff, 1), 0.5 * tf.square(absolute_diff) * classes_weights,
                                                          (absolute_diff - 0.5) * classes_weights), axis=1) * object_weights

    return tf.reduce_mean(anchorwise_smooth_l1norm, axis=0)


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

    if not skip:  Input.pre_proc_localizations(FLAGS.box_dims, thresh=0.4)

    else: print('-------------------------Previously saved records found! Loading...')

    return Input.load_protobuf(training)