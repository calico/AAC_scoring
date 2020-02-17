'''
Run regression model to score calcification

Author: Jagadish Venkataraman
Date: 4/16/2019
'''
import os.path as osp
import numpy as np
import pandas as pd
import absl.flags as flags
from .dataloader import DataLoader
from .modelbuilder import ModelBuilder

FLAGS = flags.FLAGS

def compute_aac_scores():

    config = {'data_root': osp.join(FLAGS.img_dir, 'aortic_regions'),
              'IMG_HEIGHT': FLAGS.aortic_img_height,
              'IMG_WIDTH': FLAGS.aortic_img_width,
              'backbone_network': FLAGS.backbone_network,
              'batch_size': FLAGS.batch_size,
              'model_file': FLAGS.model_file_regression,
              'mode': 'test'}

    # data DataLoader
    dl = DataLoader(config)
    print('Dataloader done')

    # model
    mb = ModelBuilder(config)
    print('Model built')

    mb.model.load_weights(config['model_file'])
    print("Loaded model from disk")
    pred_scores = mb.model.predict(dl.image_label_ds, steps=dl.image_count)
    df = pd.DataFrame(data={'img_name': [osp.basename(_) for _ in dl.all_image_paths], 'predicted_score': [np.round(np.maximum(0, p), 2) for sublist in pred_scores for p in sublist]})
    df.to_csv(osp.join(FLAGS.img_dir, 'predicted_aac_scores_model1.csv'), index=False)
    print('Predictions completed and written to predicted_aac_scores_model1.csv')

    scores_dict = {}
    img_names = [osp.basename(_) for _ in dl.all_image_paths]
    pred_scores = [np.round(np.maximum(0, p), 2) for sublist in pred_scores for p in sublist]
    for i, name in enumerate(img_names):
        scores_dict[name] = pred_scores[i]

    return scores_dict
