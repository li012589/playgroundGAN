import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def biasVariable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def discriminator(image):
    wConv1 = weightVariable([5,5,1,32])
    bConv1 = biasVariable([32])

    wConv2 = weightVariable([5,5,32,64])
    bConv2 = biasVariable([64])

    wFC1 = weightVariable([3136,1024])
    bFC1 = biasVariable([1024])

    wFC2 = weightVariable([1024,1])
    bFC2 = biasVariable([1])

    conv1 = tf.nn.conv2d(image, wConv1, strides = [1,1,1,1], padding = 'SAME') + bConv1
    conv1 = tf.nn.avg_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    conv2 = tf.nn.conv2d(conv1, wConv2, strides = [1,1,1,1], padding = 'SAME') + bConv2
    conv2 = tf.nn.avg_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    fc1 = tf.reshape(conv2,[-1,3136])
    fc1 = tf.matmul(fc1, wFC1) + bFC1
    fc1 = tf.nn.relu(fc1)

    fc2 = tf.matmul(fc1,wFC2) + bFC2

    return fc2

def generator(z, zDim):
    wFC1 = weightVariable([zDim, 3136])
    bFC1 = biasVariable([3136])

    wConv1 = weightVariable([3,3,1,zDim/2])
    bConv1 = biasVariable([zDim/2])

    wConv2 = weightVariable([3,3,zDim/2,zDim/4])
    bConv2 = biasVariable([zDim/4])

    wConv3 = weightVariable([1,1,zDim/4,1])
    bConv3 = biasVariable([1])

    fc1 = tf.matmul(z,wFC1) + bFC1
    fc1 = tf.reshape(fc1,[-1,56,56,1])
    fc1 = tf.contrib.layers.batch_norm(fc1,epsilon = 1e-5)
    fc1 = tf.nn.relu(fc1)

    conv1 = tf.nn.conv2d(fc1, wConv1, strides = [1,2,2,1], padding = 'SAME') + bConv1
    conv1 = tf.contrib.layers.batch_norm(conv1,epsilon = 1e-5)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.image.resize_images(conv1, [56, 56]) #Unpooling -transposed 2*2 avg_pooling
    #conv1 = tf.reshape(conv1,[-1,56,56,-1])

    conv2 = tf.nn.conv2d(conv1, wConv2, strides = [1,2,2,1], padding = 'SAME') + bConv2
    conv2 = tf.contrib.layers.batch_norm(conv2,epsilon = 1e-5)
    conv2 = tf.nn.relu(conv2)
    #conv2 = tf.reshape(conv2, [-1,56,56])
    conv2 = tf.image.resize_images(conv2, [56, 56])

    conv3 = tf.nn.conv2d(conv2, wConv3, strides = [1,2,2,1], padding = 'SAME') + bConv3
    conv3 = tf.sigmoid(conv3)

    return conv3

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST/")
    image = mnist.train.next_batch(1)[0].reshape([1,28,28,1])
    #print image.shape
    z = tf.placeholder(tf.float32,[None,100])
    G = generator(z,100)
    img = tf.placeholder(tf.float32,[None,28,28,1])
    D = discriminator(img)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    d = sess.run(D,feed_dict={img:image})
    tmp = sess.run(tf.random_normal([3,100],mean = 0, stddev = 1))
    g = sess.run(G,feed_dict={z:tmp})
    p1 = g[1,:,:,:].reshape([28,28])
    plt.imshow(p1)
    print d
    #images = sess.run()
    plt.show()
    while True:
        pass
