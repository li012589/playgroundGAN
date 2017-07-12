import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from ganLib import GAN

IMG_SIZE = [28,28,1]
Z_SIZE = 100
G_LEARNING_RATE = 0.0001
D_LEARNING_RATE = 0.0003

PRE_TRAIN_D = 3#300
TRIAN_TIMES = 10#1000000
BATCH_SIZE = 50

TF_SAVE_DIR = './TF_SAVE_DIR/'
PIC_SAVE_DIR = './PIC_SAVE_DIR/'
SAVE_PER_STEP = 2

def main():
    mnist = input_data.read_data_sets("MNIST/")
    sess = tf.InteractiveSession()
    gan = GAN(sess,IMG_SIZE,Z_SIZE,G_LEARNING_RATE,D_LEARNING_RATE)
    gan.init()

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(TF_SAVE_DIR)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    t = 0
    state = "pre-train"
    for i in xrange(PRE_TRAIN_D):
        t+=1
        realDate = mnist.train.next_batch(BATCH_SIZE)[0].reshape([BATCH_SIZE,IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[2]])
        gan.trainD(realDate,BATCH_SIZE)

        print("TimeStep",t,"/ State",state)
        if t % SAVE_PER_STEP == 0:
            saver.save(sess, TF_SAVE_DIR + '-gan', global_step = t)
            print("Saved")

    state = 'train'
    for i in xrange(TRIAN_TIMES):
        t+=1
        realDate = mnist.train.next_batch(BATCH_SIZE)[0].reshape([BATCH_SIZE,IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[2]])
        gan.trainG(BATCH_SIZE)

        print("TimeStep",t,"/ State",state)
        if t % SAVE_PER_STEP == 0:
            saver.save(sess, TF_SAVE_DIR + '-gan', global_step = t)
            print("Saved")

if __name__ == "__main__":
    main()
