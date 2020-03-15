'''
Create TFrecords of img/mask stacks

Author: Jagadish Venkataraman
Date: 4/16/2019
'''

from __future__ import division
import os
import os.path as osp
import glob
import random
import imageio
import numpy as np
from skimage.transform import resize
from errno import ENOENT
import tifffile
from .convert_img_mask_to_tfrecord import ConvertImgMaskToTFrecord


class CreateImageDatabase(object):
    def __init__(self, img_dir: str,
                 out_dir: str,
                 masks_dir: str=None,
                 unit_test: bool=False,
                 desired_size: tuple=(None, None)) -> None:
        '''
        Args:
            img_dir: list of input image files
            masks_dir: list of segmentation mask files
            out_dir: directory to output TF records
        '''
        self.c = ConvertImgMaskToTFrecord()
        self.img_dir = img_dir
        if masks_dir is not None:
            self.masks_dir = masks_dir
        else:
            # for inference only mode, where mask doesn't matter, the TFrecord is filled with the image itself as its mask
            self.masks_dir = img_dir
        self.out_dir = out_dir
        self.desired_size = desired_size
        self.tf_records_list = []
        self.img_mask_list = []
        self.create_img_mask_list()
        self.create_tf_records_list()
        if unit_test:
            self.do_unit_test()


    def create_img_mask_list(self) -> None:
        '''
        Look up all the mask files and the corresponding image files
        '''
        mask_files = glob.glob(osp.join(self.masks_dir, "*.png"))
        img_files = glob.glob(osp.join(self.img_dir, "*.png"))

        if not img_files:
            raise IOError(ENOENT, 'No DEXA images found, exiting')

        # ignoring image extensions, get the base names of the images
        img_base_names = [os.path.basename(name).split('.')[0] for name in img_files]
        mask_base_names = [os.path.basename(name).split('.')[0] for name in mask_files]

        # images with masks
        imgs_with_masks = list(set(img_base_names) & set(mask_base_names))

        # create img_mask list
        for name in imgs_with_masks:
            idx1 = img_base_names.index(name)
            idx2 = mask_base_names.index(name)
            self.img_mask_list.append([img_files[idx1], mask_files[idx2]])


    def create_tf_records_list(self) -> None:
        '''
        Method to convert the image/mask pair to a TF record
        '''
        print('Creating TF records')

        if not osp.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # loop over the list
        for img_file, mask_file in self.img_mask_list:
            tf_record_file = osp.join(self.out_dir, osp.basename(img_file) + '.tfrecord')
            img = self._load_img(img_file)

            if img is not None:
                mask = self._load_img(mask_file)
                self.c(img, mask, tf_record_file)
                self.tf_records_list.append(tf_record_file)


    def remove_white_corners(self, img):
        '''
        Method to remove white corners such as those in 1682184.png. Naive method that looks in the first and third
        segments of the image and eliminates max value pixels
        '''
        max_val = np.max(img)
        L = img.shape[1]//3
        img1 = img[:, :L]
        img2 = img[:, L:2*L]
        img3 = img[:, 2*L:]
        img1[np.where(img1 == max_val)] = 0
        img3[np.where(img3 == max_val)] = 0

        img_new = np.hstack([img1, img2, img3])
        return img_new


    def _load_img(self, img_file: str) -> np.ndarray:
        '''
        Method to load image and mask files based on image type and also rotate them to be of the shape (Z, H, W)

        Args:
            img_file: image file to load

        Returns:
            img: np array of shape (Z, H, W)
        '''
        if img_file.endswith('tif'):
            img = tifffile.imread(img_file)
        else:
            try:
                img = imageio.imread(img_file)
            except ValueError:
                print('Unable to read {n}, skipping...'.format(n=img_file))
                return None

        if len(img.shape) == 2:
            # if the image is grayscale, add extra dimension
            img = img[np.newaxis,...]

        # (Z,H,W) format
        if img.shape[2] < img.shape[0]:
            img = np.einsum('ijk->kij', img)

        if img.shape[0] == 4 and img_file.endswith('png'):
            # png images getting loaded as 4D images with alpha
            img = img[0,...]
            img = img[np.newaxis,...]

        if self.desired_size[0] is not None and self.desired_size[1] is not None:
            img = resize(img, tuple([img.shape[0]] + list(self.desired_size)))

        return img.astype(np.float32)


    def do_unit_test(self) -> None:
        '''
        Method to pick to a random tfrecord and perform unit testing on it by comparing its contents to the
        img and mask tensors
        '''
        test_idx = random.choice(range(len(self.tf_records_list)))
        img = self._load_img(self.img_mask_list[test_idx][0])
        mask = self._load_img(self.img_mask_list[test_idx][1])
        self.c.unit_test(img, mask, self.tf_records_list[test_idx])
