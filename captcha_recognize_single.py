from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.platform import gfile

import captcha_model as captcha
import config

IMAGE_WIDTH = config.IMAGE_WIDTH
IMAGE_HEIGHT = config.IMAGE_HEIGHT

CHAR_SETS = config.CHAR_SETS
CLASSES_NUM = config.CLASSES_NUM
CHARS_NUM = config.CHARS_NUM

FLAGS = None


def one_hot_to_texts(recog_result):
    texts = []
    for i in range(recog_result.shape[0]):
        index = recog_result[i]
        texts.append(''.join([CHAR_SETS[i] for i in index]))
    return texts


def input_data(image_path):
    if not gfile.Exists(image_path):
        print(">> Image '" + image_path + "' not found.")
        return None
    images = np.zeros([1, IMAGE_HEIGHT * IMAGE_WIDTH], dtype='float32')
    image = Image.open(image_path)
    image_gray = image.convert('L')
    image_resize = image_gray.resize(size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    image.close()
    input_img = np.array(image_resize, dtype='float32')
    images[0, :] = input_img.flatten()
    return images


def run_predict():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        input_filename = FLAGS.file_path
        input_images = input_data(input_filename)
        images = tf.math.divide(input_images, 1. / 255)
        images = tf.math.subtract(images, 0.5)
        logits = captcha.inference(images, keep_prob=1)
        result = captcha.output(logits)
        saver = tf.compat.v1.train.Saver()
        sess = tf.compat.v1.Session()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
        print(tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
        recog_result = sess.run(result)
        sess.close()
        text = one_hot_to_texts(recog_result)
        print('image ' + input_filename + " recognize ----> '" + text[0] + "'")


def main(_):
    run_predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./captcha_train',
        help='Directory where to restore checkpoint.'
    )
    parser.add_argument(
        '--file_path',
        type=str,
        help='Absolute path to the captcha image.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
