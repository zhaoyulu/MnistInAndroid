from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cv2
import time

def getPic(array,array2):

    image=np.zeros((28,28),dtype=np.uint8)

    temp = -1
    for i in range(10):
        if array2[i] == 1:
            temp = i
            break

    for i in range(28) :
        for j in range(28):
            image[i][j] = array[i*28+j]*255

    cv2.imwrite("/media/zhaoyulu/shareDisk/pic/"+(str)(int(round(time.time()*1000)))+"_"+(str)(temp)+".png", image, [int(cv2.IMWRITE_JPEG_QUALITY), 5])

for size in range(680):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    batch = mnist.train.next_batch(100)
    i = 0
    for i in range(100):
        getPic(batch[0][i],batch[1][i])