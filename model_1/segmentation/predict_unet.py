'''
Segment vertebrae and extract Aortic region for calcification scoring

Author: Jagadish Venkataraman
Date: 4/30/2019
'''

from __future__ import division
import os
import os.path as osp
import imageio
import numpy as np
import glob
import tensorflow as tf
from tf_unet import unet
import absl.flags as flags
from create_image_database import CreateImageDatabase
from extract_aortic_region_v2 import ExtractAorticRegion
from data_loader import DataLoader
import matplotlib.patches as patches
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import shutil
import cv2


FLAGS = flags.FLAGS

def predict_aortic_region():

    tfrecords_dir = osp.join(FLAGS.img_dir, 'tfrecords')
    output_dir = osp.join(FLAGS.img_dir, 'aortic_regions')

    if FLAGS.create_tfrecords:
        # create image database
        c = CreateImageDatabase(img_dir=FLAGS.img_dir, masks_dir=None, out_dir=tfrecords_dir, unit_test=False, desired_size=(FLAGS.ny, FLAGS.nx))
        filenames = c.tf_records_list
        print('TFrecords created')
    else:
        filenames = glob.glob(osp.join(tfrecords_dir, '*tfrecord'))


    if not os.path.isdir(output_dir):
        print('Creating aortic regions predictions folder...')
        os.makedirs(output_dir)
    else:
        print('Emptying existing predictions folder...')
        for f in glob.glob(osp.join(output_dir, '*')):
            os.remove(f)

    net = unet.Unet(channels=FLAGS.num_channels,
                    n_class=FLAGS.num_classes,
                    layers=FLAGS.num_layers,
                    features_root=FLAGS.num_features)
    print('Model built.')

    params = {'batch_size': FLAGS.batch_size,
              'num_classes': FLAGS.num_classes,
              'num_channels': FLAGS.num_channels,
              'repeat': False,
              'shuffle': False}

    ds = DataLoader(filenames, params)
    print('Dataloader built')

    # Aortic region extraction class
    g = ExtractAorticRegion()

    iterator = ds.dataset.make_one_shot_iterator()
    img_batch_tf, mask_batch_tf = iterator.get_next()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    counter = 0

    try:
        while True:
            img_batch, mask_batch = sess.run([img_batch_tf, mask_batch_tf])
            # convert to lists for prediction
            img_batch = img_batch[:,np.newaxis,...]
            img_list = [np.einsum('ijkl->iklj', img) for img in list(img_batch)]

            mask_batch = mask_batch[:,np.newaxis,...]
            mask_list = [np.einsum('ijkl->iklj', mask) for mask in list(mask_batch)]

            names_list = filenames[counter*FLAGS.batch_size:(counter+1)*FLAGS.batch_size]

            # predict on batch
            prediction_list, accuracy_list = net.predict_batch(FLAGS.model_file_segmentation, img_list)

            for prediction, accuracy, img, mask, name in zip(prediction_list, accuracy_list, img_list, mask_list, names_list):
                prediction = prediction[0,...]
                print('Prediction complete for {name}'.format(name=name))
                vertebrae_mask = np.zeros(prediction.shape)
                vertebrae_mask[np.where(prediction==1)] = 255

                pelvis_mask = np.zeros(prediction.shape)
                pelvis_mask[np.where(prediction==2)] = 255

                # extract aortic region
                g(img[0,...,0], vertebrae_mask, pelvis_mask, vertebrae_span=3, aorta_offset=5, aorta_width=120)

                if g.num_vertebrae > 2:
                    if FLAGS.visualize:
                        # plot the predictions
                        plt.figure(figsize=(33,6))
                        if FLAGS.masks_dir is None:
                            plt.subplot(1, 3, 1)
                            plt.imshow(img[0,...,0], cmap='Greys_r')
                            plt.title('Input')
                            plt.subplot(1, 3, 2)
                            plt.imshow(vertebrae_mask, cmap='Greys_r')
                            plt.title('Prediction - Spine')
                            plt.subplot(1, 3, 3)
                            plt.imshow(pelvis_mask, cmap='Greys_r')
                            plt.title('Prediction - Pelvis')
                            plt.tight_layout()
                            plt.savefig(osp.join(output_dir, osp.basename(name).replace('.png.tfrecord', '_pred.png')))
                        else:
                            plt.subplot(1, 4, 1)
                            plt.imshow(img[0,...,0], cmap='Greys_r')
                            plt.title('Input')
                            plt.subplot(1, 4, 2)
                            plt.imshow(mask[0,...,0], cmap='Greys_r')
                            plt.title('Ground truth')
                            plt.subplot(1, 4, 3)
                            plt.imshow(vertebrae_mask, cmap='Greys_r')
                            plt.title('Prediction - Vertebrae')
                            plt.subplot(1, 4, 4)
                            plt.imshow(pelvis_mask, cmap='Greys_r')
                            plt.title('Prediction - Pelvis')
                            plt.tight_layout()
                            plt.savefig(osp.join(output_dir, osp.basename(name).replace('.png.tfrecord', '_pred.png')))

                        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(33,6), subplot_kw={'xticks':[], 'yticks':[]})
                        ax[0].imshow(g.labeled_vertebrae)
                        x = [c[1] for c in g.vertebrae_centroids]
                        y = [c[0] for c in g.vertebrae_centroids]
                        ax[0].plot(x, y, color='red', linewidth=2)

                        if g.pelvis_centroids is not None:
                            ax[1].imshow(g.labeled_pelvis)
                            x = [c[1] for c in g.pelvis_centroids]
                            y = [c[0] for c in g.pelvis_centroids]
                            ax[1].plot([x], [y], marker='o', markersize=3, color="red")

                        ax[2].imshow(g.labeled_combine)
                        x = [c[1] for c in g.combined_centroids]
                        y = [c[0] for c in g.combined_centroids]
                        ax[2].plot(x, y, color='red', linewidth=2)
                        x = [c[1] for c in g.aorta_left]
                        y = [c[0] for c in g.aorta_left]
                        ax[2].plot(x, y, color='red', linewidth=4)
                        x = [c[1] for c in g.aorta_right]
                        y = [c[0] for c in g.aorta_right]
                        ax[2].plot(x, y, color='red', linewidth=4)
                        rect = patches.Rectangle((g.aorta_top_left_x,g.aorta_top_left_y),
                                                 (g.aorta_bottom_right_x-g.aorta_top_left_x),
                                                 (g.aorta_bottom_right_y-g.aorta_top_left_y),
                                                 linewidth=2,edgecolor='g',facecolor='none')
                        ax[2].add_patch(rect)

                        ax[3].imshow(g.aortic_region_mask, cmap='Greys_r')

                        fig.tight_layout()
                        fig.savefig(osp.join(output_dir, osp.basename(name).replace('.png.tfrecord', '_segmentations.png')))

                    # save the aortic region to file
                    image = g.aortic_region
                    # reshape and save
                    S = image.shape
                    if len(image.shape) == 2:
                        image = image[...,np.newaxis]
                    image = image[max(0, S[0]-FLAGS.aortic_img_height):min(S[0], FLAGS.aortic_img_height), max(0, S[1]-FLAGS.aortic_img_width):min(S[1], FLAGS.aortic_img_width), :]
                    S = image.shape
                    delta_w = FLAGS.aortic_img_width - S[1]
                    delta_h = FLAGS.aortic_img_height - S[0]
                    top, bottom = delta_h//2, delta_h-(delta_h//2)
                    left, right = delta_w//2, delta_w-(delta_w//2)
                    color=image[S[0]-1,0,0]
                    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
                    if len(image.shape) == 2:
                        image = image[...,np.newaxis]
                    image = (image*65535).astype(np.uint16)
                    imageio.imsave(osp.join(output_dir, osp.basename(name).replace('.png.tfrecord', '.png')), image)
                    imageio.imsave(osp.join(output_dir, osp.basename(name).replace('.png.tfrecord', '_vertebrae.png')), vertebrae_mask.astype(np.uint8))
                    imageio.imsave(osp.join(output_dir, osp.basename(name).replace('.png.tfrecord', '_pelvis.png')), pelvis_mask.astype(np.uint8))
                else:
                    # shutil.copyfile(osp.join(FLAGS.img_dir, osp.basename(name)[:-9]), osp.join(FLAGS.img_dir, 'predictions_broken', osp.basename(name)[:-9]))
                    print('Skipping image {n} with {c} vertebrae'.format(n=name,c=g.num_vertebrae))

            # batch counter
            counter += 1

    except tf.errors.OutOfRangeError:
        print('End of dataset')


if __name__ == '__main__':
    app.run(main)
