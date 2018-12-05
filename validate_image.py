import tensorflow as tf
from alexnet import AlexNet
import matplotlib.pyplot as plt
import glob
import numpy as np
import skimage.io as io
from skimage import data_dir

class_name = ['A_FFU', 'A_FL','A_NID','A_NIG','A_T','A_V']


def test_image(path_image, num_class, weights_path='Default'):

    img_string = tf.read_file(path_image)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])
    img_resized = tf.reshape(img_resized, shape=[1, 227, 227, 3])
    model = AlexNet(img_resized, 0.5, 6, skip_layer='', weights_path=weights_path)
    tf.get_variable_scope().reuse_variables()
    score = tf.nn.softmax(model.fc8)
    max = tf.arg_max(score, 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "F:/github/tensorflow_alexnet_classify_myself6/checkpoints/LR=0.001/model_epoch100.ckpt")
        # score = model.fc8
        print(sess.run(model.fc8))
        prob = sess.run(max)[0]
        print(class_name[prob])
       # plt.imshow(img_decoded.eval())
        #plt.title("Class:" + class_name[prob])
        #plt.show()


#test_image('F:/github/tensorflow_alexnet_classify_myself6/TEST_ONE/A_V__0_572.jpg', num_class=6)
train_image_path = 'F:/github/tensorflow_alexnet_classify_myself6/TEST_ONE/'
image_path= np.array(glob.glob(train_image_path + '*.jpg')).tolist()
print(len(image_path))
coll = io.ImageCollection(image_path)
print(len(coll))

for i in range(0,len(image_path)):

    img=image_path[i]
    print(img)
    test_image(img,num_class=6)
