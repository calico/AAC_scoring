'''
Outer API call to specify folder full of images to score. The script calls each of the 2 models
and stores the scores dicts independently. Finally, the scores are averaged (where possible) and
a CSV file is output with the predicted score for each image. If an image does not get a score in
either model, its score is reported as NA
'''
import pandas as pd
import argparse
import absl.flags as flags
import absl.app as app
import glob
import os
import model_1.predict_aac_scores as model_1

FLAGS = flags.FLAGS

def main(argv):

    if FLAGS.img_dir is None:
        raise ValueError('Image folder missing. Use --img_dir= argument to set this folder name.')

    # compute model 1 scores
    scores_1 = model_1.get_scores()
    # compute model 2 scores
    # scores_2 = model_2.get_scores()
    scores_2 = {}

    # get list of all png images in folder
    img_names = glob.glob(os.path.join(FLAGS.img_dir, '*png'))

    scores_dict = dict.fromkeys([os.path.basename(_) for _ in img_names], 'NA')
    # loop through the image names, check for scores and update the dict
    for img in scores_dict:
        score = -1
        if img in scores_1:
            score = scores_1[img]
        if img in scores_2:
            if score == -1:
                score = scores_2[img]
            else:
                score = (score + scores_2[img])/2

        if score != -1:
            scores_dict[img] = score

    # write to csv file
    img_names = list(scores_dict.keys())
    scores = [scores_dict[name] for name in img_names]

    df = pd.DataFrame(data={'img_name': img_names, 'predicted_score': scores})
    df.to_csv(os.path.join(FLAGS.img_dir, 'predicted_aac_scores_ensemble.csv'), index=False)
    print('Predictions completed on both models and ensemble scores written to predicted_aac_scores_ensemble.csv')


if __name__ == '__main__':
    app.run(main)
