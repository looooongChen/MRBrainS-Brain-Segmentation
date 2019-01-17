import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import tensorflow as tf
from math import pi
import random


class Dataset(object):

    def __init__(self, config):
        self.config = config
        self.subject_list = [name for name in os.listdir(config['data_dir']) if os.path.isdir(os.path.join(config['data_dir'], name))]
        self.subjects = dict((name, {}) for name in self.subject_list)
        self.shape = None

    def read_subject(self, subject_name):
        for channel, file in self.config['channels'].items():
            path = os.path.join(self.config['data_dir'], subject_name, file)
            data = nib.load(path)
            self.subjects[subject_name][channel] = {'data': data}
            if self.shape:
                assert self.shape == data.shape
            else:
                self.shape = data.shape

    def read_subjects(self, subject_list=None):
        if subject_list is None:
            subject_list = self.subjects
        for subject_name in subject_list:
            if subject_name in self.subjects.keys():
                self.read_subject(subject_name)

    def generate_ds(self, subjects, augmentation, merge_label=False):
        inputs = self.config['input_channels']
        outputs = []
        if 'output_channels' in self.config.keys():
            outputs = self.config['output_channels']
            assert len(outputs) == 1
            output_flag = True
        else:
            output_flag = False

        num, height, width = self.shape[2], self.shape[0], self.shape[1]

        input_np = np.zeros((num*len(subjects), height, width, len(inputs)))
        channels = inputs[:]
        if output_flag:
            output_np = np.zeros((num*len(subjects), height, width))
            channels.extend(outputs)

        subject_index = 0
        for subject_name in subjects:
            if subject_name in self.subject_list:
                subject = self.subjects[subject_name]
                input_index = 0
                for channel in channels:
                    data = subject[channel]['data'].get_data()
                    data = np.moveaxis(data, -1, 0)
                    if channel in inputs:
                        input_np[subject_index:subject_index + num, :, :, input_index] \
                            = (data - np.mean(data))/ np.std(data)
                        input_index += 1
                    elif channel in outputs:
                        output_np[subject_index:subject_index + num, :, :] \
                            = data
                subject_index += num
            else:
                print(subject_name + "does not exist!")

        if merge_label and output_flag:
            merge_label(output_np)

        #### data augmentation ####

        if output_flag and augmentation:
            # elatic deformation
            elastic_inputs, elastic_outputs = _elastic_transform(input_np,
                                                                 output_np,
                                                                 alpha=10, sigma=4)

            # scale
            scales = [0.9, 0.95, 1.05, 1.1]
            scaled_inputs, scaled_outputs= _scale(input_np, output_np, scales=scales)

            # translation
            with tf.device('/cpu:0'):
                translated_inputs, translated_outputs = _translate(input_np,
                                                                   output_np,
                                                                   scale=0.15, num=1)

            # rotation
            with tf.device('/cpu:0'):
                rotated_inputs, rotated_outputs = _rotate_images(input_np,
                                                                 output_np,
                                                                 start_angle=60,
                                                                 end_angle=360-60,
                                                                 n_images=5)

            #### concatenate ####

            input_np = np.concatenate((input_np, elastic_inputs), axis=0)
            output_np = np.concatenate((output_np, elastic_outputs), axis=0)

            input_np = np.concatenate((input_np, scaled_inputs), axis=0)
            output_np = np.concatenate((output_np, scaled_outputs), axis=0)

            input_np = np.concatenate((input_np, translated_inputs), axis=0)
            output_np = np.concatenate((output_np, translated_outputs), axis=0)

            input_np = np.concatenate((input_np, rotated_inputs), axis=0)
            output_np = np.concatenate((output_np, rotated_outputs), axis=0)

        if output_flag:
            return input_np, output_np.astype(np.int32)
        else:
            return input_np


def _resize(X_imgs, size, mode='label'):
    if X_imgs is None:
        return None
    shape = X_imgs.shape

    X = tf.placeholder(tf.float32, shape=(shape[0], shape[1], shape[2], shape[3]))
    tf_img = tf.image.resize_images(X, (size[0], size[1]),
                                    method=tf.image.ResizeMethod.BILINEAR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        X_resized = sess.run(tf_img, feed_dict={X: X_imgs})

    X_resized = np.array(X_resized, dtype=np.float32)
    if mode == 'label':
        X_resized = np.rint(X_resized).astype(np.uint8)
    return X_resized

def _scale(X_imgs, y_imgs, scales):

    print("==== scale for data augumentation ====")

    y_imgs = np.expand_dims(y_imgs, 3)
    inputs = np.concatenate((X_imgs, y_imgs), axis=3)
    shape = inputs.shape
    scaled = []

    X = tf.placeholder(tf.float32, (shape[1], shape[2], shape[3]))
    scaled_size = tf.placeholder(tf.int32)
    tf_scaled = tf.image.resize_images(X, scaled_size,
                                       tf.image.ResizeMethod.BILINEAR,
                                       align_corners=True)
    tf_img = tf.image.resize_image_with_crop_or_pad(tf_scaled, shape[1], shape[2])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for scale in scales:
            for img in inputs:
                scaled_img = sess.run(tf_img,
                                      feed_dict={X: img, scaled_size: (np.array(shape[1:3])*scale).astype(np.int32)})
                scaled.append(scaled_img)

    scaled = np.array(scaled, dtype=np.float32)
    X_scaled = scaled[:, :, :, :-1]
    y_scaled = np.rint(scaled[:, :, :, -1])
    y_scaled = np.squeeze(y_scaled).astype(np.uint8)

    return X_scaled, y_scaled


def _translate(X_imgs, y_imgs, scale, num=1):

    print("==== translation for data augumentation ====")

    y_imgs = np.expand_dims(y_imgs, 3)
    inputs = np.concatenate((X_imgs, y_imgs), axis=3)
    shape = inputs.shape
    translated = []

    X = tf.placeholder(tf.float32, shape=(shape[1], shape[2], shape[3]))
    d = tf.placeholder(tf.float32, shape=2)
    tf_X = tf.contrib.image.translate(X, d, interpolation='BILINEAR')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(len(inputs)):
            for _ in range(num):
                dx = int(random.uniform(-1*shape[2]*scale, shape[2]*scale))
                dy = int(random.uniform(-1*shape[1]*scale, shape[1]*scale))
                translated_img = sess.run(tf_X, feed_dict={X: inputs[i], d: np.array([dx, dy])})
                translated.append(translated_img)

    translated = np.array(translated, dtype=np.float32)
    X_translated = translated[:, :, :, :-1]
    y_translated = np.rint(translated[:, :, :, -1])
    y_translated = np.squeeze(y_translated).astype(np.uint8)

    return X_translated, y_translated


def _rotate_images(X_imgs, y_imgs, start_angle, end_angle, n_images):

    print("==== rotation for data augumentation ====")

    y_imgs = np.expand_dims(y_imgs, 3)
    inputs = np.concatenate((X_imgs, y_imgs), axis=3)
    shape = inputs.shape
    rotated = []

    iterate_at = (end_angle - start_angle) / (n_images - 1)

    X = tf.placeholder(tf.float32, shape=(shape[0], shape[1], shape[2], shape[3]))
    radian = tf.placeholder(tf.float32, shape=(len(inputs)))
    tf_img = tf.contrib.image.rotate(X, radian, interpolation='BILINEAR')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for index in range(n_images):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict={X: inputs, radian: radian_arr})
            rotated.extend(rotated_imgs)

    rotated = np.array(rotated, dtype=np.float32)
    X_rotated = rotated[:, :, :, :-1]
    y_rotated = np.rint(rotated[:, :, :, -1])
    y_rotated = np.squeeze(y_rotated).astype(np.uint8)


    return X_rotated, y_rotated




def _elastic_transform(X_imgs, y_imgs, alpha, sigma, random_state=None):

    print("==== elastic transformation for data augumentation ====")

    def op(image, label, alpha, sigma, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        sz = shape[:2]

        dx = gaussian_filter((random_state.rand(*sz) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*sz) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(sz[1]), np.arange(sz[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        transformed_image = np.zeros_like(image)

        for i in range(shape[2]):
            transformed_image[:, :, i] = map_coordinates(image[:, :, i], indices,
                                                         order=1,
                                                         mode='reflect').reshape(sz)

        transformed_label = map_coordinates(label, indices, order=1,
                                            mode='reflect').reshape(sz)

        return transformed_image, np.rint(transformed_label).astype(np.uint8)

    transform = lambda img, label: op(img, label, alpha, sigma, random_state=random_state)

    X_elas = []
    y_elas = []
    for i in range(len(X_imgs)):
        img, label = transform(X_imgs[i], y_imgs[i])
        X_elas.append(img)
        y_elas.append(label)
    return np.array(X_elas), np.array(y_elas)



def replace_label(data, ori, des):
    data[data == ori] = des


def merge_label(data):
    replace_label(data, 2, 1)
    replace_label(data, 3, 2)
    replace_label(data, 4, 2)
    replace_label(data, 5, 3)
    replace_label(data, 6, 3)
    replace_label(data, 7, 0)
    replace_label(data, 8, 0)
    replace_label(data, 9, 0)
    replace_label(data, 10, 0)

    return data

def translate_label_2013_test(data):
    # gray matter 2
    replace_label(data, 1, 2)
    # white matter 3
    replace_label(data, 4, 3)
    # CSF 1
    replace_label(data, 5, 1)
    replace_label(data, 6, 1)

    replace_label(data, 7, 0)
    replace_label(data, 8, 0)

    return data
