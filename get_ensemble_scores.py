'''
Outer API call to specify folder full of images to score. The script calls each of the 2 models
and stores the scores dicts independently. Finally, the scores are averaged (where possible) and
a CSV file is output with the predicted score for each image. If an image does not get a score in
either model, its score is reported as NA
'''
import glob
import os
import sys
sys.path.insert(0, 'model_1')
sys.path.insert(1, 'model_2')
import pandas as pd
import absl.flags as flags
import absl.app as app
import model_1.predict_aac_scores as model_1
import model_2.predict_aac_scores_2 as model_2


FLAGS = flags.FLAGS

def main(argv):

    if FLAGS.img_dir is None:
        raise ValueError('Image folder missing. Use --img_dir= argument to set this folder name.')

    # compute model 1 scores
    print('Running model 1')
    scores_1 = model_1.get_scores()
    print('Model 1 complete')
    
    # compute model 2 scores
    print('Running model 2')
    scores_2 = model_2.get_scores(FLAGS.img_dir)
    print('Model 2 complete')

    # get list of all png images in folder
    img_names = glob.glob(os.path.join(FLAGS.img_dir, '*png'))

    scores_dict = dict.fromkeys([os.path.basename(_) for _ in img_names], 'NA')
    # loop through the image names, check for scores and update the dict
    for img in scores_dict:
        if img in scores_1 and img in scores_2:
            scores_dict[img] = (scores_1[img] + scores_2[img])/2.

    # write to csv file
    img_names = list(scores_dict.keys())
    scores = [scores_dict[name] for name in img_names]

    df = pd.DataFrame(data={'img_name': img_names, 'predicted_score': scores})
    df.to_csv(os.path.join(FLAGS.img_dir, 'predicted_aac_scores_ensemble.csv'), index=False)
    print('Predictions completed on both models and ensemble scores written to {p}'.format(p=os.path.join(FLAGS.img_dir, 'predicted_aac_scores_ensemble.csv')))


if __name__ == '__main__':
    app.run(main)
