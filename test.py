
import tensorflow as tf
import scipy.io as scio
import numpy as np
from utilities import convert_test_data
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
sess = tf.InteractiveSession()#start a session
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess,ckpt.model_checkpoint_path)#load model

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
predict = tf.argmax(y_conv,1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))#compute accuracy
#testfile = 'test_32x32.mat'
#test_data = scio.loadmat(testfile)
#L = test_data['X'].shape[-1]
test_data = convert_test_data('data/test_images.mat')
L = len(test_data)
#images = test_data['X'].transpose().reshape((L,32*32*3))*1.00/255#normalization
#labels = []
all_accuracy = []
k=0
f = open('labels.txt','w+')
while(True):
    if k+100>L:
        break
    predictions = predict.eval(feed_dict={
        x: test_data[k:k+100], keep_prob: 1.0})
    for label in predictions:
      if label == 0:
        f.write(str(10)+'\n')#write result to file
      else:
        f.write(str(label)+'\n')
    #test_accuracy = accuracy.eval(feed_dict={
        #x: test_data[k:k+100], y_: labels[k:k+100], keep_prob: 1.0})#test 100 samples
    k+=100
    #all_accuracy.append(test_accuracy)
f.close()

#print('test accuracy: ',np.mean(all_accuracy))#compute whole accuracy


