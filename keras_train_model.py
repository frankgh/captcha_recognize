import tensorflow as tf

from tensorflow.keras import datasets, layers, models

IMAGE_HEIGHT = 50
IMAGE_WIDTH = 200

train_dir = '/home/fguerrero/ceac/captcha/four/train'
valid_dir = '/home/fguerrero/ceac/captcha/four/valid'

train_images = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="grayscale",
    batch_size=32,
    image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    smart_resize=False,
)

valid_images = tf.keras.preprocessing.image_dataset_from_directory(
    valid_dir,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="grayscale",
    batch_size=32,
    image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    smart_resize=False,
)