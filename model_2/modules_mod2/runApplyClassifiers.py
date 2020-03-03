#!/home/graham/anaconda2/bin/python
import os
import cv2
import sys
import argparse
import ml_tools as ML
import file_tools as FT

# THE PATHNAME TO THE MODELS APPLIED BY THIS MODULE
# is defined here.  Alternative pathways based on
# whether this script is being called directly from
# its directory or used as a module by predict_aac_scores_2.py
# from the directory above.
if os.path.basename(os.getcwd())=='modules_mod2': modPath = '../models_mod2'
else: modPath = 'models_mod2'

# This module applies three tensorflow models.  For each, there is a
# model weights file (.pb) and a model output labels file (.labels.txt).
# 
pModNameList = ["aac_bi_l3l4_low","aac_bi_l3l4_high","background_l3l4"]
pModels,pLabels = {},{}
for modName in pModNameList:
  pModels[modName] = os.path.join(modPath,modName+'.pb')
  pLabels[modName] = os.path.join(modPath,modName+'.labels.txt')
pModToAppend = {}
pModToAppend["aac_bi_l3l4_low"]  = "aacLo"
pModToAppend["aac_bi_l3l4_high"] = "aacHi"
pModToAppend["background_l3l4"]  = "back"
  

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("-i","--input_dir",
                  help="the directory of files to be evaluated")
  ap.add_argument("-o","--output_file",
                  help="file for writing the results (.tsv)")
  args = vars(ap.parse_args())
  runAnalysis(args)
  
def runAnalysis(args):
  if len(args["output_file"]) < 5:
    raise ValueError("outfile must be .tsv")
  if not(args["output_file"].split('.')[-1]=="tsv"):
    raise ValueError("outfile must be .tsv")
  outBase = '.'.join(args["output_file"].split('.')[:-1])

  for pm in pModNameList:
    classMod = ML.TfClassApplyer(pModels[pm],pLabels[pm])
    imgMang = FT.ImageIterator(args["input_dir"])
    outfName = '.'.join([outBase,pModToAppend[pm],'tsv'])
    outf = open(outfName,'w')
  
    imgMang.initiateSort()
    while imgMang.isSorting():
      infName = imgMang.getImgFile()
      imgMang.moveToNext()
      # My full-analysis pipeline will place .tsv files
      # in this directory, so I'll skip those
      if infName.split('.')[-1]!='tsv':
        img = cv2.imread(infName)
        if img is None or img.shape[0]==0 or img.shape[1]==0:
          print(infName)
          if img is None: print("\tis an empty image")
          else: print("\tis an empty image of shape "+str(img.shape))
        else:
          res = classMod.getClasses(img)
          fId = os.path.basename(infName).split('.')[0]
          outL = [fId+'.png']
          for c in res.labels():
            outL.append( (c,res.score(c)) )
          outf.write('\t'.join(map(str,outL))+'\n')
    if outf!=sys.stdout: outf.close()

if __name__ == "__main__": main()
