'''
Inference script to generate calcification scores on a folder full of DEXA CT images

Author: Jagadish Venkataraman
Date: 6/18/2019
'''
import os.path as osp
import absl.app as app
import absl.flags as flags
import sys
sys.path.insert(0, 'segmentation')
sys.path.insert(0, 'regression')
sys.path.insert(0, 'utils')
from predict_unet import predict_aortic_region
from predict_scores import compute_aac_scores
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('img_dir',
                    '/scratch/jagadish/calcification/data/segmentation/unet/data_v8/images/',
                    'directory containing images for prediction')
flags.DEFINE_string('model_file_segmentation',
                     '/scratch/jagadish/calcification/models/segmentation/unet/20190612-092624/model.ckpt-59',
                     'model file for segmentation')
flags.DEFINE_integer('num_classes',
                     3,
                     'number of output classes including background')
flags.DEFINE_integer('num_channels',
                     1,
                     '3 for RGB and 1 for grayscale input images')
flags.DEFINE_integer('nx',
                     512,
                     'width of CT image')
flags.DEFINE_integer('ny',
                     1024,
                     'height of CT image')
flags.DEFINE_integer('num_layers',
                     5,
                     'Num U-Net layers')
flags.DEFINE_integer('num_features',
                     16,
                     'Number of features in U-net')
flags.DEFINE_boolean('visualize',
                     False,
                     'When False, only Aortic region images are created. When true, visualization of segmentations are created')
flags.DEFINE_boolean('create_tfrecords',
                     False,
                     'Create TFrecords when true')
flags.DEFINE_integer('batch_size',
                     512,
                     'batch size of images to process')
flags.DEFINE_integer('aortic_img_width',
                     256,
                     'aortic region image width')
flags.DEFINE_integer('aortic_img_height',
                     256,
                     'aortic region image height')
flags.DEFINE_string('backbone_network',
                     None,
                     'Backbone network to use')
flags.DEFINE_string('model_file_regression',
                    '/scratch/jagadish/calcification/models/regression/v2/20190703-150303/checkpoints/weights.350-40.54.hdf5',
                    'model file for regression')

#tf.enable_eager_execution()


def get_scores(argv=None):

    # predict the aortic region
    predict_aortic_region()

    # compute the AAC scores
    # scores = compute_aac_scores()
    scores=None
    return scores


if __name__ == '__main__':
    app.run(get_scores)
