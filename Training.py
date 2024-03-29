
# coding: utf-8

# In[1]:

from __future__ import division
import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
from skimage import io
from skimage.util import img_as_float, img_as_ubyte
#import matplotlib.cm as cm
#from matplotlib import pyplot as plt
#%matplotlib inline
import time
from six.moves import xrange 


# In[2]:

DATA_DIR = os.path.abspath("../Data/test-ssrbc2015/segmentation")

def get_paths(directory):
    """Gets the filenames of all sclera images in the given directory along with their 
        ground truth images
        Args:
            directory: The path to the root folder
        Output:
            images: List of paths to files containing images
            ground: List of paths to files containing corresponding ground truth images
    """
    imgs = glob.glob(DATA_DIR+"/[0-9]*/E*.jpg")
    imgs.sort()
    gt = glob.glob(DATA_DIR+"/[0-9]*/M*.jpg")
    gt.sort()
    
    # The following lines of code remove those examples whose image and ground truth 
    # sizes don't match
    images = []
    ground = []
    for i in range(len(imgs)):
        img = io.imread(imgs[i])
        g = io.imread(gt[i])
        if g.shape[0] == img.shape[0] and g.shape[1] == img.shape[1]:
            images.append(imgs[i])
            ground.append(gt[i])

    return images, ground

data, ground_truth = get_paths(DATA_DIR)


# In[3]:

train_data = pd.read_pickle('../Data/test-ssrbc2015/segmentation/mn_train.pkl') 
validation_data = pd.read_pickle('../Data/test-ssrbc2015/segmentation/mn_validation.pkl') 
test_data = pd.read_pickle('../Data/test-ssrbc2015/segmentation/mn_test.pkl') 

mean_img = pd.read_pickle('../Data/test-ssrbc2015/segmentation/mean_img.pkl')


# In[4]:

# Hyper Params
TRAIN_PATCHES = len(train_data)
VALIDATION_PATCHES = len(validation_data)
TEST_PATCHES = len(test_data)
PATCHES_PER_IMAGE = 100
PATCH_DIM = 31                          # Dimension of window used as a sample
BATCH_SIZE = 60
LEARNING_RATE = 5e-4
MAX_STEPS = 200
NUM_CLASSES = 2


# In[5]:

def next_batch(size, df, current_batch_ind):
    """Returns the next mini batch of data from the dataset passed
    
    Args:
        size: length of the current requested mini batch
        df: the data set consisting of the images and the labels
        current_batch_ind: the current position of the index in the dataset
    
    Returns:
        (batch_x, batch_y): A tuple of np arrays of dimensions 
                        [size, patch_dim**2*3] and [size, NUM_CLASSES] respectively 
        current_batch_ind: the updated current position of the index in the dataset
        df: when the requested batch_size+current_batch_ind is more than the length of the data set,
            the data is shuffled again and current_batch_ind is reset to 0, and this new data set is
            returned
    
    """
    
    if current_batch_ind + size> len(df):
        current_batch_ind = 0
        df = df.iloc[np.random.permutation(len(df))]
        df = df.reset_index(drop=True)
        
    batch_x = np.zeros((size, PATCH_DIM**2*3))
    batch_y = np.zeros((size, NUM_CLASSES), dtype = 'uint8')
    for i in range(current_batch_ind, current_batch_ind+size):
        batch_x[i - current_batch_ind] = df.loc[i][:-1]
        batch_y[i - current_batch_ind][int(df.loc[i][PATCH_DIM**2*3])]=1
        
    current_batch_ind += size  
    return (batch_x, batch_y), current_batch_ind, df


# In[6]:

def inference(images, keep_prob, fc_hidden_units1=512):
    """ Builds the model as far as is required for running the network
    forward to make predictions.

    Args:
        images: Images placeholder, from inputs().
        keep_prob: Probability used for Droupout in the final Affine Layer
        fc_hidden_units1: Number of hidden neurons in final Affine layer
    Returns:
        softmax_linear: Output tensor with the computed logits.
    """
    with tf.variable_scope('h_conv1') as scope:
        weights = tf.get_variable('weights', shape=[4, 4, 3, 64], 
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.05))
        
        # Flattening the 3D image into a 1D array
        x_image = tf.reshape(images, [-1,PATCH_DIM,PATCH_DIM,3])
        z = tf.nn.conv2d(x_image, weights, strides=[1, 1, 1, 1], padding='VALID')
        h_conv1 = tf.nn.relu(z+biases, name=scope.name)
    with tf.variable_scope('h_conv2') as scope:
        weights = tf.get_variable('weights', shape=[4, 4, 64, 64], 
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.05))
        z = tf.nn.conv2d(h_conv1, weights, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2 = tf.nn.relu(z+biases, name=scope.name)
    
    h_pool1 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name='h_pool1')
    
    with tf.variable_scope('h_conv3') as scope:
        weights = tf.get_variable('weights', shape=[4, 4, 64, 64], 
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.05))
        z = tf.nn.conv2d(h_pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
        h_conv3 = tf.nn.relu(z+biases, name=scope.name)
        
    h_pool2 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')
    
    with tf.variable_scope('h_fc1') as scope:
        weights = tf.get_variable('weights', shape=[7**2*64, fc_hidden_units1], 
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=[fc_hidden_units1], initializer=tf.constant_initializer(0.05))
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights) + biases, name = 'h_fc1')
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        
    with tf.variable_scope('h_fc2') as scope:
        weights = tf.get_variable('weights', shape=[fc_hidden_units1, NUM_CLASSES], 
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=[NUM_CLASSES])
        
        logits = (tf.matmul(h_fc1_drop, weights) + biases)
    return logits


# In[7]:

def calc_loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].
    Returns:
        loss: Loss tensor of type float.
    """
    labels = tf.to_float(labels)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


# In[8]:

def training(loss, learning_rate=5e-4):
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
    # Create the Adam optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# In[9]:

def evaluation(logits, labels, topk=1):
    """Evaluate the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
                  range [0, NUM_CLASSES).
        topk: the number k for 'top-k accuracy'
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    
    correct = tf.nn.in_top_k(logits, tf.reshape(tf.slice(labels, [0,1], [int(labels.get_shape()[0]), 1]),[-1]), topk)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


# In[10]:

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
        batch_size: The batch size will be baked into both placeholders.
    Returns:
        images_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, PATCH_DIM**2*3))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size, NUM_CLASSES))
    return images_placeholder, labels_placeholder


# In[11]:

#UPDATE current_img_ind
def fill_feed_dict(data_set, images_pl, labels_pl, current_img_ind, batch_size):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
                  <placeholder>: <tensor of values to be passed for placeholder>,
                  ....
                }
    Args:
        data_set: The set of images and labels, from input_data.read_data_sets()
        images_pl: The images placeholder, from placeholder_inputs().
        labels_pl: The labels placeholder, from placeholder_inputs().
        current_img_ind: The current position of the index in the dataset
    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
        current_img_ind: The updated position of the index in the dataset
        data_set: updated data_set
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.
    batch, current_img_ind, data_set= next_batch(batch_size, data_set, current_img_ind)

    feed_dict = {
      images_pl: batch[0],
      labels_pl: batch[1],
    }
    return feed_dict, current_img_ind, data_set


# In[12]:

def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set, batch_size):
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
    steps_per_epoch = len(data_set) // batch_size
    num_examples = steps_per_epoch * batch_size
    current_img_ind = 0
    for step in xrange(steps_per_epoch):
        feed_dict, current_img_ind, data_set = fill_feed_dict(data_set, images_placeholder,
                               labels_placeholder, current_img_ind, batch_size)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            (num_examples, true_count, precision))


# In[13]:

def run_training():
    """Train for a number of steps."""
    # Tell TensorFlow that the model will be built into the default Graph.
    global train_data, test_data, validation_data
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)

        # Build a Graph that computes predictions from the inference model.
        logits = inference(images_placeholder, 0.5, 512)

        # Add to the Graph the Ops for loss calculation.
        loss = calc_loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss, LEARNING_RATE)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = evaluation(logits, labels_placeholder)

        # Build the summary operation based on the TF collection of Summaries.
        #summary_op = tf.merge_all_summaries()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        #summary_writer = tf.train.SummaryWriter('../Data/', sess.graph)
        current_img_ind = 0
        # And then after everything is built, start the training loop.
        for step in xrange(MAX_STEPS):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict, current_img_ind, train_data = fill_feed_dict(train_data,
                                 images_placeholder,
                                 labels_placeholder, current_img_ind=current_img_ind, 
                                                        batch_size=BATCH_SIZE)
            

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 1 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                #summary_str = sess.run(summary_op, feed_dict=feed_dict)
                #summary_writer.add_summary(summary_str, step)
                #summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % (50) == 0 or (step + 1) == MAX_STEPS:
                saver.save(sess, '../Data/test-ssrbc2015/segmentation/model.ckpt',global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, train_data, BATCH_SIZE)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, test_data, BATCH_SIZE)


# In[14]:

run_training()


# In[ ]:



