'''
Train Unet to segment lower vertebrae and Pelvis

Author: Jagadish Venkataraman
Date: 4/16/2019
'''

from __future__ import division
import os
import os.path as osp
import datetime
import time
import numpy as np
import absl.app as app
import absl.flags as flags
from .unet import Unet, Trainer
from .create_image_database import CreateImageDatabase
from .data_loader import DataLoader


FLAGS = flags.FLAGS
flags.DEFINE_string('img_dir',
                    None,
                    'directory containing images for training')
flags.DEFINE_string('masks_dir',
                    None,
                    'directory containing GT masks for training')
flags.DEFINE_string('logs_dir',
                     'logs/' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S'),
                     'directory to write the model and log files to')
flags.DEFINE_string('tfrecords_dir',
                     None,
                     'directory to write TF Records to for training')
flags.DEFINE_integer('batch_size',
                     8,
                     'Batch size')
flags.DEFINE_integer('shuffle_buf',
                     512,
                     'Shuffle buffer size')
flags.DEFINE_integer('num_classes',
                     3,
                     'number of output classes including background')
flags.DEFINE_integer('num_channels',
                     1,
                     '3 for RGB and 1 for grayscale input images')
flags.DEFINE_float('lr',
                   1e-4,
                   'learning rate')
flags.DEFINE_integer('num_epochs',
                     100,
                     'num training epochs')
flags.DEFINE_list('class_weights',
                  [1., 2., 2.],
                  'weights for various classes to learn')
flags.DEFINE_string('optimizer',
                     'adam',
                     'optimizer to use')
flags.DEFINE_integer('nx',
                     512,
                     'width of image')
flags.DEFINE_integer('ny',
                     1024,
                     'height of image')
flags.DEFINE_integer('num_layers',
                     5,
                     'Num U-Net layers')
flags.DEFINE_integer('num_features',
                     16,
                     'Number of features in U-net')


def main(argv):
    del argv #unused

    # create image database
    c = CreateImageDatabase(img_dir=FLAGS.img_dir,
                            out_dir=FLAGS.tfrecords_dir,
                            masks_dir=FLAGS.masks_dir,
                            unit_test=False,
                            desired_size=(FLAGS.ny, FLAGS.nx))
    filenames = c.tf_records_list

    # train and test split
    train_idx = np.random.choice(np.arange(len(filenames)),
                                 size=np.floor(0.90*len(filenames)).astype('int'),
                                 replace = False).astype('int')
    test_idx = np.setdiff1d(np.arange(len(filenames)), train_idx).astype('int')

    with open(osp.join(FLAGS.tfrecords_dir, 'train.csv'), 'w') as f:
        for name in [x for i, x in enumerate(filenames) if i in train_idx]:
            f.write("%s\n" % name)

    with open(osp.join(FLAGS.tfrecords_dir, 'test.csv'), 'w') as f:
        for name in [x for i, x in enumerate(filenames) if i in test_idx]:
            f.write("%s\n" % name)

    params = {'batch_size': FLAGS.batch_size,
                  'shuffleBuf': FLAGS.shuffle_buf,
                  'epochs': FLAGS.num_epochs,
                  'num_classes': FLAGS.num_classes,
                  'num_channels': FLAGS.num_channels}

    filenames_train = [x for i, x in enumerate(filenames) if i in train_idx]
    ds = DataLoader(filenames_train, params, augment=True)
    print('Dataloader built')

    # Model
    net = Unet(channels=FLAGS.num_channels,
                    n_class=FLAGS.num_classes,
                    layers=FLAGS.num_layers,
                    features_root=FLAGS.num_features,
                    cost_kwargs={'class_weights': FLAGS.class_weights})
    print('Model built.')

    # trainer
    trainer = Trainer(net, optimizer=FLAGS.optimizer)

    if not osp.exists(FLAGS.logs_dir):
        os.makedirs(FLAGS.logs_dir)

    # train
    path = trainer.train(ds.dataset,
                         FLAGS.logs_dir,
                         training_iters=len(filenames_train) // FLAGS.batch_size,
                         epochs=FLAGS.num_epochs,
                         display_step=2,
                         restore=True)


if __name__ == '__main__':
    app.run(main)
