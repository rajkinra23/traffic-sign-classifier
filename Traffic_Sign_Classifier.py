
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
#
# ## Deep Learning
#
# ## Project: Build a Traffic Sign Recognition Classifier
#
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary.
#
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
#
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
#
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
#
#
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[1]:


# Load pickled data
import pickle
import os

# TODO: Fill this in based on where you saved the training and testing data

# Assert that our data folder exists.
assert 'data' in os.listdir('.')
ROOT = 'data'

df = os.listdir(ROOT)
training_file = os.path.join(ROOT, df[0])
validation_file = os.path.join(ROOT, df[1])
testing_file = os.path.join(ROOT, df[2])

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# Log the shapes of our train, test, and validation sets.
print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
print(X_test.shape)
print(y_test.shape)


# ---
#
# ## Step 1: Dataset Summary & Exploration
#
# The pickled data is a dictionary with 4 key/value pairs:
#
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
#
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results.

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[2]:


### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = (X_test.shape[1], X_test.shape[2])

# TODO: How many unique classes/labels there are in the dataset.
n_classes = X_test.shape[3]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle

# Let's do two things here to preprocess our data set:

# (1) Shuffle the training examples
# (2) Normalize the colorspace from 0-255, to -0.5 to 0.5.

# (2)
normalize = lambda image: (image - 128.0)/128.0

# (1)
X_train, y_train = shuffle(X_train, y_train, random_state=1)

# (2) Normalize the datasets.
X_train = normalize(X_train)
X_valid = normalize(X_valid)
X_test = normalize(X_test)

import tensorflow as tf
from tensorflow.contrib.layers import flatten

# Define a function to build our model. LeNet, in this case.

def LeNet(x, classes, mu=0, sigma=0.1):
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = n_classes.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

# Initialize our placeholders.
x = tf.placeholder(tf.float32, (None, 32, 32, 3),  name='x')
y = tf.placeholder(tf.int32, (None), name='y')
one_hot_y = tf.one_hot(y, n_classes)

# Create the model.
logits = LeNet(x, n_classes)

# Define the cross entropy between the logits and labels.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)

# Define the loss operation, which is to reduce the mean over the cross entropy.
loss = tf.reduce_mean(cross_entropy)

# Add a log operation for the loss.
log_loss = tf.Print(loss, [loss], message="Loss: ")

# Next, create the train operation via an optimizer.
train_network = tf.train.AdamOptimizer(learning_rate=0.001).minimize(log_loss)

# Train the network here.

import numpy as np

# With 12000 examples, let's try training on 10 epochs. Let
# each batch size be length 100.
epochs = 10
batch_size = 100

# Define our prediction operation. This is just the argmax of our
# prediction being equal to the argmax ot the onehot.
is_prediction_right = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y ,1))

# The accuracy is just the reduce mean of the prediction tensor.
accuracy = tf.reduce_mean(tf.cast(is_prediction_right, tf.float32))

# Utility function for getting the batch slice from data
def get_slice(tensor, index):
    return tensor[index:index + batch_size]

# Define our eval function, using the accuracy operation. Measure the accuracy for every batch.
def evaluate_accuracy(X_tensor, y_tensor):
    # Initialize some variables.
    size = len(X_tensor)
    accuracies = []
    weights = []

    # Get the latest session.
    session = tf.get_default_session()

    # Iterate over the data using our batch size.
    for index in range(0, size, batch_size):
        x_slice, y_slice = get_slice(X_tensor, index), get_slice(y_tensor, index)
        batch_accuracy = session.run(accuracy, feed_dict={x: x_slice,
                                                          y: y_slice})

        # Add accuracy/weight to our lists.
        accuracies.append(batch_accuracy)
        weights.append(len(x_slice))

    # Use the accuracies and weights to compute a weighted average.
    total_accuracy = np.average(accuracies, weights=weights)
    return total_accuracy

# Logger function for displaying the current state.
def log_epoch(epoch, accuracy):
    print("Epoch: %s" % str(epoch))
    print("Accuracy: %s" % str(accuracy))
    print('\n')

with tf.Session() as sess:
    # Initialize global variables.
    sess.run(tf.global_variables_initializer())

    # Get the total number of examples:
    train_size = len(X_train)

    # Repeat training for every epoch.
    for epoch in range(epochs):
        # Re shuffle in each epoch.
        X_train, y_train = shuffle(X_train, y_train, random_state=epoch)

        # Per epoch, iterate over the batches with a batch size of 100.
        for index in range(0, train_size, batch_size):
            # Get X and y slices.
            X_slice, y_slice = get_slice(X_train, index), get_slice(y_train, index)

            # Feed input dict to network trainer.
            sess.run(train_network, feed_dict={x: X_slice,
                                               y: y_slice})


        # Pipe the validation data into our accuracy evaluator.
        validation_accuracy = evaluate_accuracy(X_valid, y_valid)

        # Log the outcome of each epoch.
        log_epoch(epoch, validation_accuracy)

# def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
#     # Here make sure to preprocess your image_input in a way your network expects
#     # with size, normalization, ect if needed
#     # image_input =
#     # Note: x should be the same name as your network's tensorflow data placeholder variable
#     # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
#     activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
#     featuremaps = activation.shape[3]
#     plt.figure(plt_num, figsize=(15,15))
#     for featuremap in range(featuremaps):
#         plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
#         plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
#         if activation_min != -1 & activation_max != -1:
#             plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
#         elif activation_max != -1:
#             plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
#         elif activation_min !=-1:
#             plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
#         else:
#             plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
