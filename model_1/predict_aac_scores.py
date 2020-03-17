'''
Inference script to generate calcification scores on a folder full of DEXA
CT images

Author: Jagadish Venkataraman
Date: 6/18/2019
'''
import absl.app as app
import absl.flags as flags
import os
import logging
from segmentation.predict_unet import predict_aortic_region
from regression.predict_scores import compute_aac_scores

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True

FLAGS = flags.FLAGS

flags.DEFINE_string('img_dir',
                    None,
                    'directory containing images for prediction')
flags.DEFINE_string('model_file_segmentation',
                    'model_files/final_model_unet',
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
                     'When False, only Aortic region images are created. '
                     'When true, visualization of segmentations are created')
flags.DEFINE_boolean('create_tfrecords',
                     True,
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
                    'model_files/final_model_regression.hdf5',
                    'model file for regression')


def get_scores():
    '''
    Method to compute calcification scores for the DEXA images
    '''

    # predict the aortic region
    print('Segmenting the images and extracting the Aortic regions')
    predict_aortic_region()

    # compute the AAC scores
    print('Mapping the extracted aortic regions to a score')
    scores = compute_aac_scores()
    #scores=None
    return scores


def main(argv):
    get_scores()


if __name__ == '__main__':
    app.run(main)
