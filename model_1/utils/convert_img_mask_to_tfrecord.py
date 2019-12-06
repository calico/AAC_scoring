'''
Convert Image Mask pairs to TFrecords for training

Author: Jagadish Venkataraman
Date: 4/16/2019
'''

from __future__ import division
import numpy as np
import glob
from typing import Union, Any, List, Optional, cast
import tensorflow as tf

class ConvertImgMaskToTFrecord(object):
    def __call__(self, img: np.ndarray, mask: np.ndarray, out_file: str) -> None:
        '''
        img and mask are both assumed to in the shape (Z, H, W) with Z=1 even if there are no z-slices
        '''
        self.img = self.standardize(img)
        self.mask = self.standardize(mask)
        self.out_file = out_file
        self.write_tfrecord()


    def standardize(self, img: np.ndarray) -> np.ndarray:
        '''
        Standardization of the image to lie between 0 and 1
        '''
        # standardization
        img -= np.amin(img)
        if np.amax(img) > 0:
            img /= np.amax(img)
        return img


    def _int64_feature(self, value):
        '''
        Serialize to integer list
        '''
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


    def _bytes_feature(self, value):
        '''
        Serialize to bytes list
        '''
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


    def _decode_float_array(self, string_input):
        '''
        Decode TF float array
        '''
        return tf.decode_raw(string_input, tf.float32)


    def write_tfrecord(self) -> None:
        '''
        Method to serialize the image and mask tensors into bytes lists and write them into
        a TFrecord
        '''
        feature = {'train/signal': self._bytes_feature([tf.compat.as_bytes(self.img.tostring())]), # _bytes_feature(signal),
                   'train/target': self._bytes_feature([tf.compat.as_bytes(self.mask.tostring())]), # _bytes_feature(target),
                   'train/shape':  self._int64_feature(self.img.shape),
                   'train/tarshape': self._int64_feature(self.mask.shape)}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        print("Writing: " + self.out_file)
        writer = tf.python_io.TFRecordWriter(self.out_file)
        writer.write(example.SerializeToString())
        writer.close()


    def read_from_tfrecord(self, filename: str):
        '''
        Method to deserialize a TFrecord and convert it to np arrays
        Args:
            filename: name of TFrecord file to read
        Returns:
            signal: img tensor
            target: mask tensor
            sig_shape: img shape tensor
            tar_shape: mask shape tensor
        '''
        tfrecord_file_queue = tf.train.string_input_producer(filename, name='queue')
        reader = tf.TFRecordReader()

        _, tfrecord_serialized = reader.read(tfrecord_file_queue)

        # label and image are stored as bytes but could be stored as                                             # int64 or float64 values in a serialized tf.Example protobuf.
        tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                            features = {"train/signal": tf.FixedLenFeature([], tf.string), # float32 numpy array
                    "train/target": tf.FixedLenFeature([], tf.string), # float32 numpy array
                    "train/shape": tf.VarLenFeature(tf.int64),
                    "train/tarshape": tf.VarLenFeature(tf.int64)}, name='features')

        # image was saved as uint8, so we have to decode as uint8.
        signal = self._decode_float_array(tfrecord_features['train/signal'])
        target = self._decode_float_array(tfrecord_features['train/target'])
        sig_shape = tf.sparse_tensor_to_dense(tfrecord_features['train/shape'])
        tar_shape = tf.sparse_tensor_to_dense(tfrecord_features['train/tarshape'])

        # the image tensor is flattened out, so we have to reconstruct the shape
        signal = tf.reshape(signal, sig_shape)
        target = tf.reshape(target, tar_shape)
        return signal, target, sig_shape, tar_shape


    def read_tfrecord(self, tfrecord_file: str) -> List[Union[np.ndarray, np.ndarray, tuple, tuple]]:
        print("Reading: " + tfrecord_file)
        signal, target, sig_shape, tar_shape = self.read_from_tfrecord([tfrecord_file])

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            signal, target, sig_shape, tar_shape = sess.run([signal, target, sig_shape, tar_shape])
            coord.request_stop()
            coord.join(threads)

        return signal, target


    def unit_test(self, img: np.ndarray, mask: np.ndarray, tf_record_file: str) -> None:
        '''
        Method to compare the original image and mask tensors to the ones obtained after serializing and
        deserializing.

        Args:
            img: input img tensor
            mask: input mask tensor
            tf_record_file: name of file to read
        '''
        print("Unit testing...")
        img = self.standardize(img)
        mask = self.standardize(mask)
        img_tf, mask_tf = self.read_tfrecord(tf_record_file)

        assert np.mean((img - img_tf)**2) == 0.0, "Test failed"
        assert np.mean((mask - mask_tf)**2) == 0.0, "Test failed"

        print('Passed')
