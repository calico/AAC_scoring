""" Prediction of Aortic Abdominal Calcification scores from DEXA images: method 2



written by J. Graham Ruby (2019)
"""

import modules_mod2.runScoliosisAnalysis as SAP
import modules_mod2.runApplyClassifiers as AC
import modules_mod2.runFinalRegression as FR
import argparse
import os

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("-i","--input_dir",
                  help="input directory of images to be cropped")
  ap.add_argument("-s","--storage_dir",
                  help="directory of intermediate images/results")
  ap.add_argument("-o","--output_file",
                  help="directory that models will be written to")
  ap.add_argument("--original_resize",
                  help="resizes images as in Sethi et al: REQUIRES '<ID>.png' file name format.",
                  action='store_true')
  ap.add_argument("--no_resize",
                  help="prevents images from being re-sized: only use if you've checked your image sizes against the docs!",
                  action='store_true')
  args = vars(ap.parse_args())
  runAnalysis(args)

def runAnalysis(args):
  argForSAP = {}
  argForSAP['input_dir'] = args['input_dir']
  argForSAP['output_dir'] = args['storage_dir']
  argForSAP['original_resize'] = args['original_resize']
  argForSAP['no_resize'] = args['no_resize']
  argForSAP['output_file'] = os.path.join(args['input_dir'],'predicted_scoliosis_scores.tsv')

  print("########## Extracting aortic images ################")
  SAP.runAnalysis(argForSAP)

def get_scores(imgDir):
    args = {}
    args['input_dir'] = imgDir
    tempDir = os.path.join(imgDir,'Temp')
    if not(os.path.isdir(tempDir)): os.mkdir(tempDir)
    args['storage_dir'] = tempDir
    args['output_dir'] = tempDir
    args['original_resize'] = False
    args['no_resize'] = False
    runAnalysis(args)

if __name__ == "__main__": main()


