import tensorflow as tf
#   pb to tflite
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pbfile = "/home/zhaoyulu/Documents/graph_opt.pb"
tflitefile = "/home/zhaoyulu/Desktop/mnist/mnist/mnist.tflite"

frozen_pb = pbfile
input_node_name = ['MobilenetV1/Conv2d_0/Conv2D']
output_node_name = ["Openpose/concat_stage7"]

#converter = tf.contrib.lite.TocoConverter.from_frozen_graph(frozen_pb,input_node_name,output_node_name)

converter = tf.contrib.lite.TocoConverter.from_frozen_graph(frozen_pb,
                                                            input_node_name,
                                                             output_node_name,
                                                             input_shapes={
                                                                          "input":[1,224,224,3]
                                                                           }
                                                            )

tflite_model = converter.convert()
open(tflitefile, "wb").write(tflite_model)
print("Generate tflite success.")
