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

    # Inputs = batchx64x64x64
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
    conv = sdn.residual_layer('Conv4d', conv, 3, K * 8, 1, phase_train=phase_train) # Smaller net here
    conv = sdn.residual_layer('Conv4e', conv, 3, K * 8, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv4f', conv, 3, K * 8, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv4g', conv, 3, K * 8, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv4h', conv, 3, K * 8, 1, phase_train=phase_train)

    # At this point split into classifier and regressor
    convC = sdn.inception_layer('ConvC1', conv, K * 16, S=2, phase_train=phase_train)  # 4
    convC = sdn.residual_layer('ConvC2', convC, 3, K * 16, 1, phase_train=phase_train)
    convC = sdn.residual_layer('ConvC3', convC, 3, K * 16, 1, phase_train=phase_train)
    convC = sdn.residual_layer('ConvC4', convC, 3, K * 16, 1, phase_train=phase_train)
    linearC = sdn.fc7_layer('FC7c', convC, 8, True, phase_train, FLAGS.dropout_factor, override=3, BN=True)
    linearC = sdn.linear_layer('LinearC', linearC, 4, True, phase_train, FLAGS.dropout_factor, BN=True)
    LogitsC = sdn.linear_layer('SoftmaxC', linearC, 2, relu=False, add_bias=False, BN=False)

    if FLAGS.net_type == 'RPN':
        # Regressor, aka predict how many pixels to move over to get to center
        convR = sdn.inception_layer('ConvR1', conv, K * 16, S=2, phase_train=phase_train)  # 4
        convR = sdn.residual_layer('ConvR2', convR, 3, K * 16, 1, phase_train=phase_train)
        convR = sdn.residual_layer('ConvR3', convR, 3, K * 16, 1, phase_train=phase_train)
        convR = sdn.residual_layer('ConvR4', convR, 3, K * 16, 1, phase_train=phase_train)
        linearR = sdn.fc7_layer('FC7r', convR, 8, True, phase_train, FLAGS.dropout_factor, override=3, BN=True)
        LogitsR = sdn.linear_layer('SoftmaxR', linearR, 4, relu=False, add_bias=False, BN=False)

    # Return logits as a tuple
    if FLAGS.net_type == 'RPN': return (LogitsC, LogitsR)
    else: return (LogitsC, LogitsC)


def total_loss(logits, labels):

    """
    Add loss to the trainable variables and a summary
    logits is a tuple of Classification logits and regression logits
    Saved the data to [0ymin, 1xmin, 2ymax, 3xmax, cny, cnx, 6height, 7width, 8origshapey, 9origshapex,
        10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, 18IOU, 19obj_class, 20#_class]
    """

    # Loss factors: 1e2, 0.0 and 2.0 lead to nan. 1e2, 1e-3 and 1.0 lead to nana
    class_loss_factor = 1e2
    loc_loss_factor = 1e-4
    foreground_class_weight = 2.0

    # Squish
    labels = tf.cast(tf.squeeze(labels), tf.float32)
    logitsC = tf.squeeze(logits[0])

    """
    Regression loss, activate only for true anchors and normalize by that number
    The goal is to learn to predict how many pixels to move over to get to center of the scaphoid and how much to scale the anchor window by
    So predict the difference between the applied transformation 
    And the factor to scale the anchor box window by
    """

    if FLAGS.net_type == 'RPN':

        logitsR = tf.squeeze(logits[1])

        # Make an object mask for the localization loss
        object_mask = tf.cast(labels[:, 19] > 0, tf.float32)

        # Calculate localization deltas required to actually shift the box
        ceny, cenx, h, w = tf.unstack(labels[:, 4:8], axis=1)
        cenya, cenxa, ha, wa = tf.unstack(labels[:, 14:18], axis=1)

        # Avoid NaN in division and log below.
        ha += 1e-8
        wa += 1e-8
        h += 1e-8
        w += 1e-8

        # Calculate the normalized translations ACTUALLY required to shift the bbox
        tx, ty = (cenx - cenxa) / wa, (ceny - cenya) / ha
        tw, th = tf.log(w / wa), tf.log(h / ha)
        actual = tf.transpose(tf.stack([ty, tx, th, tw]))

        # Get loss for each translation
        absolute_diff = tf.cast(tf.abs(logitsR - actual), tf.float32)
        loc_loss = tf.reduce_sum(tf.where(tf.less(absolute_diff, 1), 0.5 * tf.square(absolute_diff), absolute_diff - 0.5), axis=1) * object_mask

        # Normalize by number of objects included here
        loc_loss = tf.divide(tf.reduce_sum(loc_loss), tf.reduce_sum(object_mask))*loc_loss_factor/4

    """
    Classification loss
    """
    # Change labels to one hot
    labelsC = tf.one_hot(tf.cast(labels[:, 19], tf.uint8), depth=2, dtype=tf.uint8)

    # # Make a weighting mask for object foreground classification loss
    # class_mask = tf.cast(labels[:, 19] > 0, tf.float32)
    #
    # # Now multiply this mask by scaling factor then add back to labels. Add 1 to prevent 0 loss
    # class_mask = tf.add(tf.multiply(class_mask, foreground_class_weight), 1)
    #
    # # Calculate  loss
    # class_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.squeeze(labelsC), logits=logitsC)
    #
    # # Add in classification mask
    # if foreground_class_weight != 1.0: class_loss = tf.multiply(class_loss, tf.squeeze(class_mask))
    #
    # # Reduce to scalar
    # class_loss = tf.reduce_sum(class_loss)

    # Use focal loss, 1 or 2
    # class_loss = focal_loss(logitsC, labelsC)
    class_loss = tf.reduce_sum(focal_softmax_cross_entropy_with_logits(labelsC, logitsC))

    # Normalize by minibatch size
    class_loss = class_loss_factor*tf.divide(class_loss, FLAGS.batch_size)

    """
    Combine Losses
    """

    # Add losses
    if FLAGS.net_type == 'RPN': total_loss = class_loss + loc_loss
    else: total_loss = class_loss

    # Output the summary of the MSE and MAE
    tf.summary.scalar('Class_Loss', class_loss)
    if FLAGS.net_type == 'RPN': tf.summary.scalar('Loc_Loss', loc_loss)
    tf.summary.scalar('Tot_Loss', total_loss)

    # Add these losses to the collection
    tf.add_to_collection('losses', class_loss)
    if FLAGS.net_type == 'RPN': tf.add_to_collection('losses', loc_loss)
    tf.add_to_collection('losses', total_loss)

    if FLAGS.net_type == 'RPN': return total_loss, class_loss, loc_loss
    else: return total_loss, class_loss, class_loss


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

    # clip the gradients
    gradients = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gradients]

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


from tensorflow.python.ops import array_ops
def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    target_tensor = tf.cast(target_tensor, tf.float32)
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)

def focal_softmax_cross_entropy_with_logits(
        labels, logits, focus=2.0, alpha=0.25, eps=1e-7, name='focal_softmax_cross_entropy_with_logits'):
    with tf.name_scope(name):
        alpha = tf.multiply(tf.cast(tf.ones_like(labels), tf.float32), alpha)

        prob = tf.nn.softmax(logits, -1)

        labels_eq_1 = tf.equal(labels, 1)
        a_balance = tf.where(labels_eq_1, alpha, 1 - alpha)
        prob_true = tf.where(labels_eq_1, prob, 1 - prob)
        modulating_factor = (1.0 - prob_true)**focus

        log_prob = tf.log(prob + eps)
        loss = -tf.reduce_sum(a_balance * modulating_factor * tf.cast(labels, tf.float32) * log_prob, -1)

        return loss