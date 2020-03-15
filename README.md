# Scoring of the abdominal aortic calcification

This repository automates the abdominal aortic calcification scoring of the DEXA images in the UKBB dataset. The scoring is based on the scheme shown below (from [this](https://bmcnephrol.biomedcentral.com/articles/10.1186/s12882-017-0480-2) reference paper).

![ScreenShot](model_1/images/Abdominal_aortic_calcification_quantification.png)

Given a folder full of DEXA images from the UKBB dataset (after the images have been converted to png format from the DICOM format), the code can be run to generate scores only using model_1 (U-net for segmentation + aortic region extraction + regression for scoring), only using model_2 or to generate ensemble scores from model_1 and model_2.

The ensemble scores for model_1 and model_2 are computed for the DEXA images in a folder. A sample inference call is as follows:

```
python get_ensemble_scores.py --img_dir=<absolute path to folder containing DEXA images>
```

The ensemble scores are output in a csv file called 'predicted_aac_scores_ensemble.csv' in the folder containing the DEXA images.

**Note**
You will need access to the DEXA images from the UK BioBank dataset to be able to run the models presented in this repository. All modeling choices are tailored to these images.
