from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse
from os import path
import sys

import tensorflow as tf
import captcha_model as captcha
from tensorflow.python.client import device_lib

FLAGS = None


def run_train():
    """Train CAPTCHA for a number of steps."""

    with tf.Graph().as_default():
        images, labels = captcha.inputs(train=True, batch_size=FLAGS.batch_size)

        logits = captcha.inference(images, keep_prob=0.5)

        loss = captcha.loss(logits, labels)

        train_op = captcha.training(loss)

        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

        init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                           tf.compat.v1.local_variables_initializer())

        sess = tf.compat.v1.Session()
        sess.run(init_op)
        initial_step = 0

        print(device_lib.list_local_devices())

        if path.exists(FLAGS.checkpoint_dir):
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
            print('')
            print('')
            print('=======================================================')
            print('=======================================================')
            print('=                                                     =')
            print('=                                                     =')
            print('=                                                     =')
            print('=  Loading from ' +FLAGS.checkpoint_dir + '           =')
            print('=                                                     =')
            print('=                                                     =')
            print('=                                                     =')
            print('=======================================================')
            print('=======================================================')
            print('')
            print('')
            print(tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
            last_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            initial_step = int(last_checkpoint[last_checkpoint.rfind('-') + 1:])
        else:
            os.mkdir(FLAGS.checkpoint_dir)

        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
        try:
            step = initial_step
            while not coord.should_stop():
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time
                if step % 10 == 0:
                    print('>> Step %d run_train: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                                            duration))
                if step != initial_step and step % 100 == 0:
                    print('>> %s Saving in %s' % (datetime.now(), FLAGS.checkpoint))
                    saver.save(sess, FLAGS.checkpoint, global_step=step)
                step += 1
        except Exception as e:
            print('>> %s Saving in %s' % (datetime.now(), FLAGS.checkpoint))
            saver.save(sess, FLAGS.checkpoint, global_step=step)
            coord.request_stop(e)
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()


def main(_):
    if tf.io.gfile.exists(FLAGS.train_dir):
        tf.io.gfile.rmtree(FLAGS.train_dir)
    tf.io.gfile.makedirs(FLAGS.train_dir)
    run_train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size.'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='./captcha_train',
        help='Directory where to write event logs.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./captcha_train/captcha',
        help='Directory where to write checkpoint.'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./captcha_train',
        help='Directory where to restore checkpoint.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
