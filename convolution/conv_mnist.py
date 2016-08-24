import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.Session()


# define graph variables
x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, shape=[-1, 28, 28, 1])
target = tf.placeholder(tf.float32, shape=[None, 10])
w1 = tf.Variable(tf.truncated_normal(shape=[5,5,1,32], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[32]))
w2 = tf.Variable(tf.truncated_normal(shape=[5,5,32,64], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[64]))
w3 = tf.Variable(tf.truncated_normal(shape=[7*7*64, 1024]))
b3 = tf.Variable(tf.constant(0.1, shape=[1024]))
keep_prob = tf.placeholder(tf.float32)
w4 = tf.Variable(tf.truncated_normal(shape=[1024, 10]))
b4 = tf.Variable(tf.constant(0.1, shape=[10]))

# define graph layers
h1_conv = tf.nn.relu(tf.nn.conv2d(x_image, w1, strides=[1,1,1,1], padding='SAME') + b1)
h1_pool = tf.nn.max_pool(h1_conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
h2_conv = tf.nn.relu(tf.nn.conv2d(h1_pool, w2, strides=[1,1,1,1], padding='SAME') + b2)
h2_pool = tf.nn.max_pool(h2_conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
h2_pool_flat = tf.reshape(h2_pool, shape=[-1, 7*7*64])
h1_fc = tf.nn.relu(tf.matmul(h2_pool_flat, w3) + b3)
# h1_dropout = tf.nn.dropout(h1_fc, keep_prob=keep_prob)
output = tf.nn.softmax(tf.matmul(h1_fc, w4) + b4)

# define loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(target * tf.log(tf.clip_by_value(output, 1e-10, 1.0)), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(target,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

# define training loop
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        weights = sess.run(w1)
        print(weights)
        train_accuracy = sess.run(accuracy, feed_dict={
            x:batch[0], target: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch[0], target: batch[1], keep_prob: 1.0})

print("test accuracy %g"%sess.run(accuracy, feed_dict={
    x: mnist.test.images, target: mnist.test.labels, keep_prob: 1.0}))




