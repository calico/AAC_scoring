'''
Dataloader class for the U-Net

Author: Jagadish Venkataraman
Date: 4/16/2019
'''

from __future__ import division
import numpy as np
import tensorflow as tf

class DataLoader(object):
    def __init__(self, filenames, params=None, seed=None, augment=False):
        self.filenames = filenames
        self.params = params
        self.dataset = None
        if seed is None:
            self.seed = np.random.randint(0, 1000)
        else:
            self.seed = seed
        self._data_loader(augment)


    def _corrupt_brightness(self, image, mask):
        """
        Applies a random brightness change.
        """
        image = tf.image.random_hue(image, 0.1)
        return image, mask


    def _corrupt_contrast(self, image, mask):
        """
        Applies a random contrast change.
        """
        image = tf.image.random_contrast(image, 0.2, 1.8)
        return image, mask


    def _corrupt_saturation(self, image, mask):
        """
        Applies a random saturation change.
        """
        image = tf.image.random_saturation(image, 0.2, 1.8)
        return image, mask


    def _flip_left_right(self, image, mask):
        """
        Flips image and mask left or right in accord.
        """

        def _left_right(img):
            img = tf.einsum('ijk->jki', img)
            img = tf.image.random_flip_left_right(img, seed=self.seed)
            img = tf.einsum('ijk->kij', img)
            return img

        image = _left_right(image)
        mask = _left_right(mask)

        return image, mask


    def _flip_up_down(self, image, mask):
        """
        Flips image and mask up or down in accord.
        """
        def _up_down(img):
            img = tf.einsum('ijk->jki', img)
            img = tf.image.random_flip_up_down(img, seed=self.seed)
            img = tf.einsum('ijk->kij', img)
            return img

        image = _up_down(image)
        mask = _up_down(mask)

        return image, mask


    def _rotate_90(self, image, mask):
        """
        Rotates image by + 90 degrees
        """
        def _rotate(img):
            img = tf.einsum('ijk->jki', img)
            img = tf.image.rot90(img, k=1)
            img = tf.einsum('ijk->kij', img)
            return img

        image = _rotate(image)
        mask = _rotate(mask)

        return image, mask


    def _rotate_270(self, image, mask):
        """
        Rotates image by - 270 degrees
        """
        def _rotate(img):
            img = tf.einsum('ijk->jki', img)
            img = tf.image.rot90(img, k=3)
            img = tf.einsum('ijk->kij', img)
            return img

        image = _rotate(image)
        mask = _rotate(mask)

        return image, mask


    def _decode_float_array(self, string_input):
        '''
        Decode TF float array
        '''
        return tf.decode_raw(string_input, tf.float32)


    def _parse_function(self, example_proto):

        tfrecord_features = tf.io.parse_single_example(example_proto,
                            features = {"train/signal": tf.io.FixedLenFeature([], tf.string),
                    "train/target": tf.io.FixedLenFeature([], tf.string),
                    "train/shape": tf.io.VarLenFeature(tf.int64),
                    "train/tarshape": tf.io.VarLenFeature(tf.int64)}, name='features')

        # image was saved as uint8, so we have to decode as uint8.
        signal = self._decode_float_array(tfrecord_features['train/signal'])
        target = self._decode_float_array(tfrecord_features['train/target'])
        sig_shape = tf.sparse.to_dense(tfrecord_features['train/shape'])
        tar_shape = tf.sparse.to_dense(tfrecord_features['train/tarshape'])

        # the image tensor is flattened out, so we have to reconstruct the shape
        signal = tf.reshape(signal, sig_shape)
        target = tf.reshape(target, tar_shape)

        return signal, target


    def _data_loader(self, augment=False):
        print('Augment: {val}'.format(val=augment))
        # extract params
        batch_size = self.params.get('batch_size', 8)
        shuffle_buf = self.params.get('shuffleBuf', 4)
        repeat = self.params.get('repeat', True)
        shuffle = self.params.get('shuffle', True)

        # dataset API
        self.dataset = tf.data.TFRecordDataset(self.filenames)
        if shuffle:
            self.dataset = self.dataset.shuffle(shuffle_buf)
        self.dataset = self.dataset.map(lambda e: self._parse_function(e), num_parallel_calls=32)

        if augment:
            augmented_set = self.dataset.map(self._flip_left_right,
                            num_parallel_calls=32)
            self.dataset = self.dataset.concatenate(augmented_set)

            augmented_set = self.dataset.map(self._flip_up_down,
                            num_parallel_calls=32)
            self.dataset = self.dataset.concatenate(augmented_set)

        if repeat:
            self.dataset = self.dataset.repeat()

        self.dataset = self.dataset.batch(batch_size, drop_remainder=False).prefetch(32)
