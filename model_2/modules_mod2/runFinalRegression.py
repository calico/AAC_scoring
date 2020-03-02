import numpy as np, os, argparse
import keras
from keras.models import load_model

# get the machine scores
def mkCatScrDD(fname,colStart=1):
  """
  Extracts the scores assigned for each class for each
  image from a properly-formatted image-classificaiton
  results file.
  REQUIRED FORMAT: tab-deliminted text.
  Col 0: image name/ID
  Cols 1+: for each column, a python-interpretable
    tuple of the following format: "('<name>', <score>)"
    where <name> is the name of a class (all classes should
    be represented, but order doesn't matter), and <score>
    is the probability score returned for that class.  Note
    the single quotes around <name>, indicating it to be a
    string.
  NOTE that additional data can be stored in columns 1-colStart
    if that arg is used (in the past, I've used it for easily marking
    which class got the highest score).
  """
  csDD = {}
  f = open(fname)
  tL = map(lambda i: i.rstrip(), f.readlines())
  f.close()
  for n in range(len(tL)):
    c = tL[n].split('\t')
    id = c[0].split('.')[0]
    csDD[id] = {}
    for i in c[colStart:]:
      j = i[1:-1].split(', ')
      k = j[0][1:-1]
      v = float(j[1])
      csDD[id][k] = v
  return csDD

# keys: model names; values: files w/results
def getDataArray(modToF):
  """
  Collects the ouput probabilities from the arg-specified
  files and puts them into a data array of appropriate structure
  for the regression model that will be applied to it.  This is
  NOT a general-use function: it is specific to the structure and
  nature of input data for the regression model, and it implements
  that specificity in its expectation of specific keys in "modToF".
  """
  modToDD = {}
  for m in ['background','high_aac','low_aac']:
    modToDD[m] = mkCatScrDD(modToF[m])
  # validate that the same IDs are there for each model
  idL = modToDD['background'].keys()
  for m in ['high_aac','low_aac']:
    if len(modToDD[m]) != len(idL):
      raise ValueError('different # scores for '+m)
    for i in idL:
      if not(modToDD[m].has_key(i)):
        raise ValueError('ID "'+i+'" was missing from '+m)
  # get the data arrays, in proper order (see nb146 p67)
  dataL = []
  for i in idL:
    dL = [modToDD['background'][i]['low'],
          modToDD['background'][i]['medium'],
          modToDD['background'][i]['high'],
          modToDD['low_aac'][i]['calc'],
          modToDD['high_aac'][i]['calc']   ]
    dataL.append(dL)
  dataA = np.array(dataL)
  return dataA,idL

# THE PATHNAME TO THE MODEL APPLIED BY THIS MODULE
# is defined here.  Alternative pathways based on
# whether this script is being called directly from
# its directory or used as a module by predict_aac_scores_2.py
# from the directory above.
if os.path.basename(os.getcwd())=='modules_mod2': modPath = '../models_mod2'
else: modPath = 'models_mod2'
regressModFile = os.path.join(modPath,"final_regress.h5")

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("-o","--output_file",
                  help="file that scores will be written to")
  ap.add_argument("--data_bg",
                  help="input data from application of the background noise model")
  ap.add_argument("--data_high",
                  help="input data from application of the high-threshold calc model")
  ap.add_argument("--data_low",
                  help="input data from application of the low-threshold calc model")
  args = vars(ap.parse_args())
  runAnalysis(args)
  
def runAnalysis(args):
  argToMod = {}
  argToMod['data_low'] = 'low_aac'
  argToMod['data_high'] = 'high_aac'
  argToMod['data_bg'] = 'background'
  
  model = load_model(regressModFile)
  model.summary()
  modToF = {}
  for a in argToMod.keys():
    modToF[argToMod[a]] = args[a]
  inputData,idL = getDataArray(modToF)
  print "got data"
  
  outputData = model.predict(inputData).flatten()
  print "applied model"
  
  fo = open(args["output_file"],'w')
  for n in range(len(idL)):
    fo.write(idL[n]+'\t'+str(outputData[n])+'\n')
    pass # write the file-then-score output format
  fo.close()
  print "wrote file"


if __name__ == "__main__": main()
