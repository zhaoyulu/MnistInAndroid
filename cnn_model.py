from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from PIL import Image

try:
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter


#读取数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess=tf.InteractiveSession()
#构建cnn网络结构
#自定义卷积函数（后面卷积时就不用写太多）
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
#自定义池化函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def avage_pool_2x2(y):
    return tf.nn.avg_pool(y, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')


#设置占位符，尺寸为样本输入和输出的尺寸
x=tf.placeholder(tf.float32,[None,784],name = "addData")
y_=tf.placeholder(tf.float32,[None,10])
x_img=tf.reshape(x,[-1,28,28,1],name = "input")

#设置第一个卷积层和池化层
w_conv1=tf.Variable(tf.truncated_normal([3,3,1,64],stddev=0.1))
b_conv1=tf.Variable(tf.constant(0.1,shape=[64]))
h_conv1=tf.nn.relu(conv2d(x_img,w_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

#设置第二个卷积层和池化层
w_conv2=tf.Variable(tf.truncated_normal([3,3,64,64],stddev=0.1))
b_conv2=tf.Variable(tf.constant(0.1,shape=[64]))
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)


#设置第三个卷积层和池化层
w_conv3=tf.Variable(tf.truncated_normal([3,3,64,64],stddev=0.1))
b_conv3=tf.Variable(tf.constant(0.1,shape=[64]))
h_conv3=tf.nn.relu(conv2d(h_pool2,w_conv3)+b_conv3)
h_pool3=avage_pool_2x2(h_conv3)


#dropout（随机权重失活）
keep_prob=tf.placeholder(tf.float32)
#h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#设置第二个全连接层
w_fc2=tf.Variable(tf.truncated_normal([7*7*64,10],stddev=0.1))
b_fc2=tf.Variable(tf.constant(0.1,shape=[10]))
h_pool2_flat=tf.reshape(h_pool3,[-1,7*7*64])
y_out=tf.nn.softmax(tf.matmul(h_pool2_flat,w_fc2)+b_fc2,name = "output")

#建立loss function，为交叉熵
loss=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_out),reduction_indices=[1]))
loss_summary = scalar_summary('loss', loss)

#配置Adam优化器，学习速率为1e-4
train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)

#建立正确率计算表达式
correct_prediction=tf.equal(tf.argmax(y_out,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
acc_summary = scalar_summary('accuracy', accuracy)


merged = merge_summary([loss_summary, acc_summary])
#开始喂数据，训练
tf.global_variables_initializer().run()
saver = tf.train.Saver()

writer =  SummaryWriter('/home/zhaoyulu/Desktop/mnist/log/', sess.graph)
counter = 0
for i in range(100):

    batch=mnist.train.next_batch(60)
    for k in range(1):
        temp = batch[k][0]
        for j in range(len(temp)):
            if temp[j] < 0.45:
                temp[j] = 0
            else:
                temp[j] =1

    if i%100==0:
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1})
        print("step %d,train_accuracy= %g"%(i,train_accuracy))

    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

    summary, _ = sess.run([merged, train_step], feed_dict={x:batch[0],y_:batch[1]})
    counter += 1
    writer.add_summary(summary, counter)


saver.save(sess, "/home/zhaoyulu/Desktop/mnist/mnist/mnist.ckpt")#w
constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output'])
with tf.gfile.GFile("/home/zhaoyulu/Desktop/mnist/mnist/wsj.pb", mode='wb') as f:
    f.write(constant_graph.SerializeToString())


#训练之后，使用测试集进行测试，输出最终结果
print("test_accuracy= %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1}))
