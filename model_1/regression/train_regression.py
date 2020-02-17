'''
Train regression model to score calcification

Author: Jagadish Venkataraman
Date: 4/16/2019
'''
import datetime
import time
import tensorflow as tf
import absl.app as app
import absl.flags as flags
from .dataloader import DataLoader
from .modelbuilder import ModelBuilder

FLAGS = flags.FLAGS
flags.DEFINE_string('data_root',
                    '/scratch/jagadish/calcification/data/segmentation/unet/inference/png_October2019_subset/aortic_regions/',
                    'directory containing images for training')
flags.DEFINE_string('gt_csv_file',
                     'median_score_gt.csv',
                     'CSV file containing median calcification scores')
flags.DEFINE_string('logdir',
                     '/scratch/jagadish/calcification/models/regression/v2/' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S'),
                     'log directory')
flags.DEFINE_string('backbone_network',
                     None,
                     'Backbone network to use')
flags.DEFINE_integer('batch_size',
                     16,
                     'batch size')
flags.DEFINE_integer('IMG_WIDTH',
                     256,
                     'image width')
flags.DEFINE_integer('IMG_HEIGHT',
                     256,
                     'image height')
flags.DEFINE_integer('epochs',
                     350,
                     'epochs to train')
flags.DEFINE_float('lr',
                   0.001,
                   'initial learning rate')


def main(argv):
    del argv #unused

    tf.enable_eager_execution()

    config = {'gt_csv_file': FLAGS.gt_csv_file,
              'data_root': FLAGS.data_root,
              'mode': 'train',
              'IMG_HEIGHT': FLAGS.IMG_HEIGHT,
              'IMG_WIDTH': FLAGS.IMG_WIDTH,
              'backbone_network': FLAGS.backbone_network,
              'batch_size': FLAGS.batch_size,
              'logdir': FLAGS.logdir,
              'lr': FLAGS.lr}

    # data DataLoader
    dl = DataLoader(config)
    print('Dataloader done')

    # model
    mb = ModelBuilder(config)
    print('Model built')

    steps_per_epoch = dl.train_size//FLAGS.batch_size
    validation_steps = dl.val_size//FLAGS.batch_size

    mb.model.fit(dl.train_ds.repeat(),
              steps_per_epoch = steps_per_epoch,
              epochs=FLAGS.epochs,
              initial_epoch = 0,
              validation_data=dl.val_ds.repeat(),
              validation_steps=validation_steps,
              callbacks=[mb.tensorboard_callback,
              mb.model_checkpoint_callback,
              mb.early_stopping_checkpoint,
              mb.lrate])


if __name__ == '__main__':
    app.run(main)
