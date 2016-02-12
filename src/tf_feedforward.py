import numpy as np

import tensorflow as tf
import time

from sklearn import cross_validation
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
from tensorflow.models.rnn.translate import data_utils
import math


def inference(input_size, output_size, features, hidden1_units, hidden2_units):
    """Build the MNIST model up to where it may be used for inference.

    Args:
      features: Images placeholder, from inputs().
      hidden1_units: Size of the first hidden layer.
      hidden2_units: Size of the second hidden layer.

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([input_size, hidden1_units],
                                stddev=1.0 / math.sqrt(float(input_size))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(features, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, output_size],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([output_size]),
                             name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits


def calculate_loss(output_size, logits, labels):
    """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
    # Convert from sparse integer labels in the range [0, NUM_CLASSSES)
    # to 1-hot dense float vectors (that is we will have batch_size vectors,
    # each with NUM_CLASSES values, all of which are 0.0 except there will
    # be a 1.0 in the entry corresponding to the label).
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, output_size]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                            onehot_labels,
                                                            name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def training(loss, learning_rate):
    """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Size is not batch_size !!!

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def do_eval(sess,
            eval_correct,
            features_placeholder,
            labels_placeholder,
            features, labels, batch_size):
    """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
    # And run one epoch of eval.

    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = features.shape[0] // batch_size
    num_examples = steps_per_epoch * batch_size
    start = 0
    for step in xrange(steps_per_epoch):
        end = start + batch_size
        feed_dict = {
            features_placeholder: features[start:end],
            labels_placeholder: labels[start:end],
        }
        start = end
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = 1.0 * true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def _data_with_context(context_size, input_size, feat_lab):
    X = []
    y = []
    for (xs, ys) in feat_lab:
        for i in xrange(context_size, len(xs) - context_size):
            X.append(
                np.ndarray(shape=(input_size,), dtype=float, buffer=xs[i - context_size: i + 1 + context_size]))
            y.append(ys[i])
    X = np.array(X)
    y = np.array(y)

    return X, y


def train_and_test(alphabet, feat_labs):
    """Train and test my feedforward neural net for a number of steps."""
    context_size = 7

    feat_len = np.shape(feat_labs[0][0])[1]
    input_size = feat_len * (2 * context_size + 1)
    output_size = len(alphabet)

    random_split = cross_validation.ShuffleSplit(len(feat_labs), n_iter=1, test_size=0.3)
    for train_index, test_index in random_split:
        train_X, train_Y = _data_with_context(context_size, input_size, feat_labs[train_index])
        test_X, test_Y = _data_with_context(context_size, input_size, feat_labs[test_index])

    batch_size, learning_rate = 128, 0.003

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the data and labels.
        features_placeholder = tf.placeholder(tf.float32, shape=(batch_size, train_X.shape[1]))
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

        # Build a Graph that computes predictions from the inference model.
        logits = inference(input_size, output_size, features_placeholder, 1024, 1024)

        # Add to the Graph the Ops for loss calculation.
        loss = calculate_loss(output_size, logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss, learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = evaluation(logits, labels_placeholder)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter("report", graph_def=sess.graph_def)

        # And then after everything is built, start the training loop.
        n_steps = 3000
        for step in xrange(n_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            start = (step * batch_size) % train_X.shape[0]
            end = start + batch_size
            if end > train_X.shape[0]:
                # Shuffle the data
                perm = np.arange(train_X.shape[0])
                np.random.shuffle(perm)
                train_X = train_X[perm]
                train_Y = train_Y[perm]
                # Start next epoch
                start = 0
                end = start + batch_size
            feed_dict = {
                features_placeholder: train_X[start:end],
                labels_placeholder: train_Y[start:end],
            }

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 20 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == n_steps:
                saver.save(sess, "report", global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        features_placeholder,
                        labels_placeholder,
                        train_X, train_Y, batch_size)
                # # Evaluate against the validation set.
                # print('Validation Data Eval:')
                # do_eval(sess,
                #         eval_correct,
                #         features_placeholder,
                #         labels_placeholder,
                #         data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        features_placeholder,
                        labels_placeholder,
                        test_X, test_Y, batch_size)
