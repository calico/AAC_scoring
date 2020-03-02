""" Prediction of Aortic Abdominal Calcification scores from DEXA images: method 2



written by J. Graham Ruby (2019)
"""

import modules_mod2.runMakeAortaPics as MAP
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
  argForMAP = {}
  argForMAP['input_dir'] = args['input_dir']
  argForMAP['output_dir'] = args['storage_dir']
  argForMAP['original_resize'] = args['original_resize']
  argForMAP['no_resize'] = args['no_resize']

  argForAC = {}
  argForAC['input_dir'] = args['storage_dir']
  argForAC['output_file'] = os.path.join(args['storage_dir'],'results.tsv')

  argForFR = {}
  argForFR['data_low'] = os.path.join(args['storage_dir'],'results.aacLo.tsv')
  argForFR['data_high'] = os.path.join(args['storage_dir'],'results.aacHi.tsv')
  argForFR['data_bg'] = os.path.join(args['storage_dir'],'results.back.tsv')
  argForFR['output_file'] = args['output_file']

  print "########## Extracting aortic images ################"
  MAP.runAnalysis(argForMAP)
  print "########## Applying classification models ##########"
  AC.runAnalysis(argForAC)
  print "##########Applying regression model ################"
  FR.runAnalysis(argForFR)

# for use by the two-model script
def get_scores(imgDir):
    argForMAP = {}
    argForMAP['input_dir'] = imgDir
    tempDir = os.path.join(imgDir,'Temp')
    if not(os.path.isdir(tempDir)): os.mkdir(tempDir)
    argForMAP['output_dir'] = tempDir
    argForMAP['original_resize'] = False
    argForMAP['no_resize'] = False
    runAnalysis(argForMAP)
    raise ValueError("make dictionary")
  
if __name__ == "__main__": main()


