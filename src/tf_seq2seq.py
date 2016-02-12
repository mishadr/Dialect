import numpy as np
import os
import sys
import tensorflow as tf
import time
from sklearn import cross_validation
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
from tensorflow.models.rnn.translate import data_utils
from tensorflow.models.rnn.translate import seq2seq_model
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn
from tensorflow.python.framework import dtypes


def inference(feature_size, decoder_vocab_size, feature_placeholders, output_placeholders,
              info_placeholder, feed_previous, reuse_codecs=None):
    """Build the seq2seq model up to where it may be used for inference.
    Args:
    Returns:
      computed logits (predictions).
    """

    # Firstly we should map speech features (which is 128-vectors of floats) to RNN-friendly inputs
    # (should be ints). It is possible to use maxout network (like in http://arxiv.org/pdf/1412.1602.pdf,
    # where it was preliminary trained with HMMs and tuned).

    # Assuming batch_size = 1
    # Emulating get_batch() and step() from seq2seq_model.py

    # I'd like to prune inputs and outputs according to their actual lengths, but I don't know the way to do it
    input_size = len(feature_placeholders)
    output_size = len(output_placeholders)

    # encoder inputs are just reversed list of features
    encoder_inputs = list(reversed(feature_placeholders[0:input_size]))

    # decoder inputs are list of labels with symbol 'GO' prepended, transformed to one-hot form.
    # FIXME GO_ID = 1, what if label class = 1 too?
    # Assuming one-hot form
    decoder_inputs = output_placeholders[0:output_size]

    # run basic seq2seq encoder-decoder
    size = 300
    cell_enc_in = rnn_cell.LSTMCell(num_units=size, input_size=feature_size)
    cell_internal = rnn_cell.LSTMCell(num_units=size, input_size=size)
    cell_dec_in = rnn_cell.LSTMCell(num_units=size, input_size=decoder_vocab_size)
    cell_dec_out = rnn_cell.LSTMCell(num_units=size, input_size=size, num_proj=decoder_vocab_size)

    encoder_cell = rnn_cell.MultiRNNCell([cell_enc_in, cell_internal])
    decoder_cell = rnn_cell.MultiRNNCell([cell_dec_in, cell_dec_out])
    # outputs, states = seq2seq.basic_rnn_seq2seq(encoder_inputs, decoder_inputs,
    #                                             rnn_cell.MultiRNNCell([cell0] + [cell] * 2))

    with vs.variable_scope("custom_rnn_seq2seq", reuse=reuse_codecs):
        _, enc_states = rnn.rnn(encoder_cell, encoder_inputs, dtype=dtypes.float32)
        outputs, states = seq2seq.rnn_decoder(decoder_inputs, enc_states[-1], decoder_cell)

    return outputs


def calculate_loss(output_size, logits, labels_placeholders, decoder_vocab_size):
    """Calculates the loss from the logits and the labels.
    """
    # Now I should calculate loss analogously to sequence_loss() from seq2seq_model.py

    # Our targets are labels sequence (equals output sequence shifted by one).
    targets = labels_placeholders[0:output_size]

    # target weights are 1s except the last
    target_weights = [tf.ones(shape=[1], dtype=tf.float32) for _ in xrange(len(targets))]
    target_weights[-1] = tf.zeros(shape=[1], dtype=tf.float32)

    loss = seq2seq.sequence_loss(logits, targets, target_weights, decoder_vocab_size, )
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


def to_one_hot(classes, output_size):
    res = []
    for c in np.array(classes):
        array = np.zeros((output_size,), dtype=np.float32)
        array[c - 1] = 1
        res.append(array)
    return np.array(res)


def evaluation(feature_size, decoder_vocab_size, inputs_placeholders, output_placeholders,
               info_placeholder, labels_placeholders):
    """Evaluate the quality of the logits at predicting the label.
    """

    logits = inference(feature_size, decoder_vocab_size, inputs_placeholders, output_placeholders,
                       info_placeholder, False, True)
    # # For a classifier model, we can use the in_top_k Op.
    # # It returns a bool tensor with shape [batch_size] that is true for
    # # the examples where the label's is was in the top k (here k=1)
    # # of all logits for that example.
    # correct = tf.nn.in_top_k(logits, labels_placeholders, 1)
    # # Return the number of true entries.
    # return tf.reduce_sum(tf.cast(correct, tf.int32))

    return [tf.argmax(logit, tf.constant(1, tf.int32)) for logit in logits]
    # return tf.zeros(shape=[1], dtype=dtypes.int32)


def train_and_test(alphabet, feat_labs):
    """Train and test my simple seq2seq model."""

    # Extending alphabet with special symbols
    end = len(alphabet)
    alphabet[data_utils._UNK] = 0
    alphabet[data_utils._GO] = end + data_utils.GO_ID
    alphabet[data_utils._PAD] = end + data_utils.PAD_ID
    alphabet[data_utils._EOS] = end + data_utils.EOS_ID

    feature_size = np.shape(feat_labs[0][0])[1]
    decoder_vocab_size = len(alphabet)

    random_split = cross_validation.ShuffleSplit(len(feat_labs), n_iter=1, test_size=0.3)
    for train_index, test_index in random_split:
        train_set = feat_labs[train_index]
        test_set = feat_labs[test_index]

    max_inputs = np.max([len(xs) for xs, _ in feat_labs])
    max_outputs = np.max([len(ys) for _, ys in feat_labs])
    learning_rate = 0.05

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the data and labels.
        inputs_placeholders = [tf.placeholder(tf.float32, shape=(1, feature_size), name="input_{0}".format(i))
                               for i in xrange(max_inputs)]
        output_placeholders = [tf.placeholder(tf.float32, shape=(1, decoder_vocab_size), name="output_{0}".format(i))
                               for i in xrange(max_outputs + 1)]
        labels_placeholders = [tf.placeholder(tf.int32, shape=(1), name="label_{0}".format(i))
                               for i in xrange(max_outputs + 1)]
        info_placeholder = tf.placeholder(tf.int32, shape=(2), name="info")

        def prepare_data(xs, ys):
            # Input feed: encoder inputs, decoder inputs, labels and actual lengths of inputs and output.
            inputs = [np.reshape(feat_vec, (1, feature_size)) for feat_vec in xs]
            outputs = [np.reshape(v, (1, decoder_vocab_size)) for v in
                       to_one_hot(np.append(ys, alphabet[data_utils._EOS]), decoder_vocab_size)]
            labels = [np.reshape(l, (1,)) for l in np.append(ys, alphabet[data_utils._EOS])]
            # print "inputs: %s, outputs: %s, labels: %s" %\
            #       (str(np.shape(inputs)), str(np.shape(outputs)),str(np.shape(labels)))
            input_dict = {pl: data for pl, data in

                          zip(inputs_placeholders,
                              inputs + (len(inputs_placeholders) - len(inputs)) * [np.zeros(shape=(1, feature_size))])
                          + zip(output_placeholders,
                                outputs + (len(output_placeholders) - len(outputs)) * [np.zeros(shape=(1, decoder_vocab_size))])
                          + zip(labels_placeholders,
                                labels + (len(labels_placeholders) - len(labels)) * [np.zeros(shape=(1), dtype=np.int32)])}
            input_dict[info_placeholder] = (len(inputs), len(outputs))
            return input_dict

        def do_eval(sess, eval_correct, test_set):
            """Runs one evaluation against the full epoch of data.
            """

            true_count = 0  # Counts the number of correct predictions.
            num_examples = len(test_set)
            for xs, ys in test_set:
                input_feed = prepare_data(xs, ys)
                # true_count += sess.run(eval_correct, feed_dict=input_feed)
                res = sess.run(eval_correct, feed_dict=input_feed)
                res = list(np.resize(res, (len(res))))
                if alphabet[data_utils._EOS] in res:
                    res = res[:res.index(alphabet[data_utils._EOS])]
                print res, ys

            precision = 1.0 * true_count / num_examples
            print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
                  (num_examples, true_count, precision))


        # Build a Graph that computes predictions from the inference model.
        logits = inference(feature_size, decoder_vocab_size, inputs_placeholders, output_placeholders,
                           info_placeholder, True)

        # Add to the Graph the Ops for loss calculation.
        loss = calculate_loss(max_outputs+1, logits, labels_placeholders, decoder_vocab_size)
        print('1')

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss, learning_rate)
        print('2')

        # # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = evaluation(feature_size, decoder_vocab_size, inputs_placeholders, output_placeholders,
                                  info_placeholder, labels_placeholders)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        sess.run(tf.initialize_all_variables())

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter("report", graph_def=sess.graph_def)
        print('3')

        # And then after everything is built, start the training loop.
        n_steps = 3000
        for step in xrange(n_steps):
            start_time = time.time()

            # print('Starting step %d' % step)
            # Fill a feed dictionary with the actual set of features and labels (i.e. those for one utterance)
            # for this particular training step.
            xs, ys = train_set[step % len(train_set)]
            input_feed = prepare_data(xs, ys)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=input_feed)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 1 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary_op, feed_dict=input_feed)
                summary_writer.add_summary(summary_str, step)

                # Save a checkpoint and evaluate the model periodically.
                if (step+1) % 100 == 0 or (step + 1) == n_steps:
                    saver.save(sess, "report", global_step=step)

                    # Evaluate against the training set.
                    print('Training Data Eval:')
                    do_eval(sess, eval_correct, train_set)

                    # Evaluate against the test set.
                    print('Test Data Eval:')
                    do_eval(sess, eval_correct, test_set)
