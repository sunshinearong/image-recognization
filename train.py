import tensorflow as tf
import scipy.io as scio
import numpy as np
from utilities import changedataformat
def weight_variable(shape):#initial weight for layer
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):#initial bias for layer
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):#create conventional layer
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')#keep same length for output and input

def max_pool_2x2(x):#create max pooling
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
x = tf.placeholder("float", shape=[None, 32*32*3])#define input
y_ = tf.placeholder("float", shape=[None, 10])#define output
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])#define filter
x_image = tf.reshape(x, [-1,32,32,3])#reshape the input data
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#computional node in the graph
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#second conventional layer
h_pool2 = max_pool_2x2(h_conv2)
W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])#densely connected layer

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)#dropout make model more robust
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#use gradient descent to update parameters
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#correct_prediction = tf.equal(y_conv,y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))#compute accuracy
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
j=0
labels,data = changedataformat('/data/train.mat')
for i in range(20000):
    if j+50 > 73257:
        j=0
    #batch_x = data['X'][:,:,:,j:j+50].transpose().reshape(50,32*32*3)*1.00/255#normolization
    #batch_y = labels[j:j+50]#batch size is 50
    batch_x = data[j:j+50]
    batch_y = labels[j:j+50]
    j += 50
    if i%100 == 0:#print train accuracy every 100 times
        train_accuracy = accuracy.eval(feed_dict={
            x:batch_x, y_: batch_y, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

saver = tf.train.Saver(tf.trainable_variables())
saver.save(sess,'Save.ckpt',global_step= 1)#save model,should change
#saver = tf.train.Saver(tf.all_variables())
#saver.save(sess,'Save.ckpt')#save parameters,should change
