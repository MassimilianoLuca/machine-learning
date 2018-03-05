import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from array import array
import math


# https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data
def next_batch(num, data, labels):

    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
# labels are not considered
def next_batch_test(num, data):
    for i in range(0, math.ceil(len(data) / num)):
        yield data[i*num : i *num+ num]
# from jupyter notebook
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
# from jupyter notebook
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
# from jupyter notebook
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=False)
# from jupyter notebook
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

X_train = genfromtxt('train-data.csv', delimiter=',')
y_pre = genfromtxt('train-target.csv',delimiter=',', dtype="|U5")
X_test = genfromtxt('test-data.csv', delimiter=',')

y_pre = [[ord(x)-ord('a')] for x in y_pre]


encoder = OneHotEncoder()
encoder.fit(y_pre)
y_train = encoder.transform(y_pre).toarray()

x = tf.placeholder(tf.float32, [None, 128])
W = tf.Variable(tf.zeros([128, 26]))
b = tf.Variable(tf.zeros([26]))
y = tf.placeholder(tf.float32, [None, 26])

y_hat = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

image = tf.reshape(x, [-1, 16, 8, 1])
#L1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#L2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#FC1
W_fc1 = weight_variable([4 * 2 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 4*2*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#FC2
W_fc2 = weight_variable([1024, 26])
b_fc2 = bias_variable([26])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_hat = tf.nn.softmax(y_conv)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

for i in range(2000):
    batchX, batchY = next_batch(200,X_train,y_train)
    if i % 200 == 0:
        train_acc = accuracy.eval(feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
        # only for report
        #print(train_acc)
    train_step.run(feed_dict={x: batchX, y: batchY, keep_prob: 0.5})

predictions = np.array([], dtype=np.dtype("int32"))
for batch_data in next_batch_test(100,X_test):
    batch_predictions = sess.run(tf.argmax(y_hat, 1), feed_dict={x: batch_data, keep_prob: 1.0})
    predictions = np.append(predictions, batch_predictions)

results=[]
for prediction in predictions:
    letter=(chr(prediction+ord('a')))
    results.append(letter);

results_stripped = str.join("\n", results)
with open("test-targets.txt", "w") as f:
    f.write(results_stripped)
