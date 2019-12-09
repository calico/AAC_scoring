'''
Dataloader for regression model

Author: Jagadish Venkataraman
Date: 4/16/2019
'''
import numpy as np
import model_1.utils.tifffile as tifffile
import pandas as pd
import tensorflow as tf
import os.path as osp
import cv2
import glob
from tensorflow.keras.applications.resnet50 import preprocess_input as pre_res
from tensorflow.keras.applications.inception_v3 import preprocess_input as pre_iv3


class DataLoader(object):
    def __init__(self, config):
        self.gt_csv_file = config.get('gt_csv_file', None)
        self.data_root = config.get('data_root', None)
        self.mode = config.get('mode', 'train')
        self.IMG_HEIGHT = config.get('IMG_HEIGHT', 256)
        self.IMG_WIDTH = config.get('IMG_WIDTH', 256)
        self.backbone_network = config.get('backbone_network', None)
        self.num_channels = config.get('num_channels', 1)
        self.batch_size = config.get('batch_size', 16)

        if self.mode == 'train':
            self.augment = True
        else:
            self.augment = False

        self.get_images_and_labels()
        self.generate_dataset()


    def get_images_and_labels(self):
        '''
        Method to load the images and labels
        '''
        if self.mode == 'train':
            df = pd.read_csv(osp.join(self.data_root, self.gt_csv_file))
            temp_all_image_paths = df['img_name']
            for idx, val in enumerate(temp_all_image_paths):
                temp_all_image_paths[idx] = osp.join(self.data_root, val)

            temp_all_image_labels = [float(v) for v in df['score']]
        else:
            temp_all_image_paths = glob.glob(osp.join(self.data_root, '*png'))
            temp_all_image_labels = [0. for _ in temp_all_image_paths]

        # remove missing filenames
        pop_list = []
        all_image_labels_dict = {}
        for img, label in zip(temp_all_image_paths, temp_all_image_labels):
            all_image_labels_dict[img] = label
            if not osp.isfile(img):
                pop_list.append(img)
        for img in pop_list:
            all_image_labels_dict.pop(img)

        self.all_image_paths = list(all_image_labels_dict.keys())
        self.all_image_labels = [all_image_labels_dict[img] for img in self.all_image_paths]
        self.image_count = len(self.all_image_labels)

        print('Loaded a total of {n} images'.format(n=self.image_count))


    def preprocess_image_resnet(self, image):
        '''
        Method to pre-process for a resnet backbone
        '''
        image = tf.image.decode_png(image)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_image_with_crop_or_pad(image, self.IMG_HEIGHT, self.IMG_WIDTH)
        image = tf.image.grayscale_to_rgb(image)
        return image


    def preprocess_image_iv3(self, image):
        '''
        Method to pre-process for an inception backbone
        '''
        image = tf.image.decode_png(image)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_image_with_crop_or_pad(image, self.IMG_HEIGHT, self.IMG_WIDTH)
        image = tf.image.grayscale_to_rgb(image)
        return image


    def preprocess_image(self, image):
        '''
        Method to pre-process for simple conv-net backbone
        '''
        image = tf.image.decode_png(image)
        image = tf.cast(image, tf.float32)
        return image


    def get_real_median(self, v):
        v = tf.reshape(v, [-1])
        l = v.get_shape()[0]
        mid = l//2 + 1
        val = tf.nn.top_k(v, mid).values
        if l % 2 == 1:
            return val[-1]
        else:
            return 0.5 * (val[-1] + val[-2])


    def standardize_image(self, image, label):
        '''
        Method to standardize the images for different models
        '''
        if self.backbone_network is None:
            image -= tf.math.reduce_min(image)
            #image -= self.get_real_median(image)
            image = tf.math.maximum(0, tf.cast(image, tf.int32))
            image /= tf.math.reduce_max(image)
        elif self.backbone_network == 'resnet':
            image = pre_res(image)
        elif self.backbone_network == 'inception':
            image = pre_iv3(image)
        else:
            raise ValueError("Unknown backbone network specified")

        return image, label


    def _corrupt_brightness(self, image, label):
        """
        Applies a random brightness change.
        """
        image = tf.cond(label >= 1, lambda: tf.image.random_brightness(image,0.3), lambda: image)
        return image, label


    def _corrupt_contrast(self, image, label):
        """
        Applies a random contrast change.
        """
        image = tf.cond(label >= 2, lambda: tf.image.random_contrast(image, 0.2, 1.8), lambda: image)
        return image, label


    def _flip_left_right(self, image, label):
        """
        Flips image left or right.
        """

        def _left_right(img):
            img = tf.einsum('ijk->jki', img)
            img = tf.image.random_flip_left_right(img)
            img = tf.einsum('ijk->kij', img)
            return img

        image = tf.cond(label >= 3, lambda: _left_right(image), lambda: image)

        return image, label


    def _flip_up_down(self, image, label):
        """
        Flips image up or down.
        """
        def _up_down(img):
            img = tf.einsum('ijk->jki', img)
            img = tf.image.random_flip_up_down(img)
            img = tf.einsum('ijk->kij', img)
            return img

        image = tf.cond(label >= 4, lambda: _up_down(image), lambda: image)

        return image, label


    def load_and_preprocess_image(self, path, label):
        image = tf.io.read_file(path)
        return self.preprocess_image(image), label


    def _set_shapes(self, image, label):
        '''
        Method to set shapes for the tensors
        '''
        image.set_shape([self.IMG_HEIGHT, self.IMG_WIDTH, self.num_channels])
        return image, label


    def generate_dataset(self):
        '''
        Method to generate the dataset with augmentation if needed
        '''

        path_ds = tf.data.Dataset.from_tensor_slices((self.all_image_paths, self.all_image_labels))
        self.image_label_ds = path_ds.map(self.load_and_preprocess_image)
        self.image_label_ds = self.image_label_ds.map(self._set_shapes)

        # augment
        if self.augment:
            augmented_ds = self.image_label_ds.map(self._corrupt_brightness)
            self.image_label_ds = self.image_label_ds.concatenate(augmented_ds)
            augmented_ds = self.image_label_ds.map(self._corrupt_contrast)
            self.image_label_ds = self.image_label_ds.concatenate(augmented_ds)
            augmented_ds = self.image_label_ds.map(self._flip_left_right)
            self.image_label_ds = self.image_label_ds.concatenate(augmented_ds)

            self.image_count *= 8

        # standardize
        self.image_label_ds = self.image_label_ds.map(self.standardize_image)

        if self.mode == 'train':
            self.image_label_ds = self.image_label_ds.shuffle(self.image_count)
            # train/val split
            self.train_size = int(0.8 * self.image_count)
            self.val_size = self.image_count - self.train_size

            self.train_ds = self.image_label_ds.take(self.train_size)
            self.val_ds = self.image_label_ds.skip(self.train_size)

            self.train_ds = self.train_ds.batch(self.batch_size)
            self.train_ds = self.train_ds.prefetch(32)

            self.val_ds = self.val_ds.batch(self.batch_size)
            self.val_ds = self.val_ds.prefetch(32)
        else:
            self.image_label_ds = self.image_label_ds.batch(self.batch_size)
            self.image_label_ds = self.image_label_ds.prefetch(32)
