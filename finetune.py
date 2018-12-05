"""
writen by stephen
"""

import os
import numpy as np
import tensorflow as tf
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
import glob
import matplotlib.pyplot as plt#画图专用
from tensorflow.contrib.data import Iterator

plt.rcParams['font.sans-serif']=['SimHei']#显示中文不乱码
plt.rcParams['axes.unicode_minus'] = False#显示中文不乱码

learning_rate = 1e-3
num_epochs = 100 # 代的个数
batch_size = 256 #1024
dropout_rate = 0.5
num_classes = 6  # 类别标签
train_layers = ['fc8', 'fc7', 'fc6','pool5','conv5']
#train_layers = ['fc8', 'fc7', 'fc6']
#train_layers = ['fc8']
display_step = 20

filewriter_path = "F:/github/tensorflow_alexnet_classify_myself6/tensorboard"  # 存储tensorboard文件
checkpoint_path = "F:/github/tensorflow_alexnet_classify_myself6/checkpoints/LR=0.001"  # 训练好的模型和参数存放目录

if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

train_image_path = 'F:/github/tensorflow_alexnet_classify_myself6/train_image/'  # 指定训练集数据路径（根据实际情况指定训练数据集的路径）
test_image_A_FFU_path = 'F:/github/tensorflow_alexnet_classify_myself6/test_image/A_FFU/'  # 指定测试集数据路径（根据实际情况指定测试数据集的路径）
test_image_A_FL_path = 'F:/github/tensorflow_alexnet_classify_myself6/test_image/A_FL/' # 指定测试集数据路径（根据实际情况指定测试数据集的路径）
test_image_A_NID_path = 'F:/github/tensorflow_alexnet_classify_myself6/test_image/A_NID/'
test_image_A_NIG_path = 'F:/github/tensorflow_alexnet_classify_myself6/test_image/A_NIG/'
test_image_A_T_path = 'F:/github/tensorflow_alexnet_classify_myself6/test_image/A_T/'
test_image_A_V_path = 'F:/github/tensorflow_alexnet_classify_myself6/test_image/A_V/'

label_path = []
test_label = []

# 打开训练数据集目录，读取全部图片，生成图片路径列表
image_path= np.array(glob.glob(train_image_path + 'A_FFU.*.jpg')).tolist()
image_path_A_FL = np.array(glob.glob(train_image_path + 'A_FL.*.jpg')).tolist()
image_path_A_NID = np.array(glob.glob(train_image_path + 'A_NID.*.jpg')).tolist()
image_path_A_NIG = np.array(glob.glob(train_image_path + 'A_NIG.*.jpg')).tolist()
image_path_A_T = np.array(glob.glob(train_image_path + 'A_T.*.jpg')).tolist()
image_path_A_V = np.array(glob.glob(train_image_path + 'A_V.*.jpg')).tolist()

image_path [len(image_path):len(image_path_A_FL)] = image_path_A_FL
image_path [len(image_path):len(image_path_A_NID)] = image_path_A_NID
image_path [len(image_path):len(image_path_A_NIG)] = image_path_A_NIG
image_path [len(image_path):len(image_path_A_T)] = image_path_A_T
image_path [len(image_path):len(image_path_A_V)] = image_path_A_V

for i in range(len(image_path)):
    if 'A_FFU' in image_path[i]:
        label_path.append(5)
    elif 'A_FL' in image_path[i]:
        label_path.append(4)
    elif 'A_NID' in image_path[i]:
        label_path.append(3)
    elif 'A_NIG' in image_path[i]:
        label_path.append(2)
    elif 'A_T' in image_path[i]:
        label_path.append(1)
    else:
        label_path.append(0)

        ######PYTHON循环语句,不确定






# 打开测试数据集目录，读取全部图片，生成图片路径列表
test_image = np.array(glob.glob(test_image_A_FFU_path + '*.jpg')).tolist()
test_image_path_A_FL = np.array(glob.glob(test_image_A_FL_path + '*.jpg')).tolist()
test_image_path_A_NID = np.array(glob.glob(test_image_A_NID_path + '*.jpg')).tolist()
test_image_path_A_NIG = np.array(glob.glob(test_image_A_NIG_path + '*.jpg')).tolist()
test_image_path_A_T= np.array(glob.glob(test_image_A_T_path + '*.jpg')).tolist()
test_image_path_A_V= np.array(glob.glob(test_image_A_V_path + '*.jpg')).tolist()

test_image[len(test_image):len(test_image_path_A_FL)] = test_image_path_A_FL
test_image[len(test_image):len(test_image_path_A_NID)] = test_image_path_A_NID
test_image[len(test_image):len(test_image_path_A_NIG)] = test_image_path_A_NIG
test_image[len(test_image):len(test_image_path_A_T)] = test_image_path_A_T
test_image[len(test_image):len(test_image_path_A_V)] = test_image_path_A_V

for i in range(len(test_image)):
    if i < 133:
        test_label.append(5)
    elif 132<i<195:
        test_label.append(4)
    elif 194<i<239:
        test_label.append(3)
    elif 238<i<275:
        test_label.append(2)
    elif 274<i<315:
        test_label.append(1)
    else:
        test_label.append(0)


# 调用图片生成器，把训练集图片转换成三维数组
tr_data = ImageDataGenerator(
    images=image_path,
    labels=label_path,
    batch_size=batch_size,
    num_classes=num_classes)

# 调用图片生成器，把测试集图片转换成三维数组
test_data = ImageDataGenerator(
    images=test_image,
    labels=test_label,
    batch_size=batch_size,
    num_classes=num_classes,
    shuffle=False)
with tf.name_scope('input'):
    # 定义迭代器
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                   tr_data.data.output_shapes)

    training_initalize=iterator.make_initializer(tr_data.data)
    testing_initalize=iterator.make_initializer(test_data.data)

    # 定义每次迭代的数据
    next_batch = iterator.get_next()

x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# 图片数据通过AlexNet网络处理
model = AlexNet(x, keep_prob, num_classes, train_layers)

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# 执行整个网络图
score = model.fc8

with tf.name_scope('loss'):
    # 损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                              labels=y))

gradients = tf.gradients(loss, var_list)

gradients = list(zip(gradients, var_list))

with tf.name_scope('optimizer'):
    # 优化器，采用梯度下降算法进行优化
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)



# 定义网络精确度
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 把精确度加入到Tensorboard
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver()

# 定义一代的迭代次数
train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 把模型图加入Tensorboard
    writer.add_graph(sess.graph)

    # 把训练好的权重加入未训练的网络中
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # 总共训练10代
    for epoch in range(num_epochs):
        sess.run(training_initalize)
        print("{} Epoch number: {} start".format(datetime.now(), epoch + 1))

        #开始训练每一代
        for step in range(train_batches_per_epoch):
            img_batch, label_batch = sess.run(next_batch)
            sess.run(train_op, feed_dict={x: img_batch,
                                           y: label_batch,
                                           keep_prob: dropout_rate})
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch * train_batches_per_epoch + step)

        # 测试模型精确度
        print("{} Start validation".format(datetime.now()))
        sess.run(testing_initalize)
        test_acc = 0.
        test_count = 0


        for _ in range(test_batches_per_epoch):
            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.0})


            test_acc += acc
            test_count += 1




        test_acc /= test_count



        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))


        # 把训练好的模型存储起来
        print("{} Saving checkpoint of model...".format(datetime.now()))

        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Epoch number: {} end".format(datetime.now(), epoch + 1))

        plt.plot(epoch+1, test_acc,'g_')  # 调用pylab的plot函数绘制曲线
        plt.xlabel('迭代次数')
        plt.ylabel('准确率')
        #plt.title('train process figure')
    plt.show()


