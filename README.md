# AAC_scoring

This repository automates the abdominal aortic calcification scoring of the DEXA images in the UKBB dataset. The scoring is based on the scheme shown below (from [this](https://bmcnephrol.biomedcentral.com/articles/10.1186/s12882-017-0480-2) reference paper).

![ScreenShot](images/Abdominal_aortic_calcification_quantification.png)

## Inference

Given a folder full of DEXA images from the UKBB dataset (after some postprocessing steps - clarify with Anurag before publishing), the code can be run to generate scores only using model_1 (U-net for segmentation + aortic region extraction + regression for scoring) or to generate ensemble scores from model_1 and model_2.

### Mode 1
In this mode, only model_1 scores are computed for the images in the folder. A sample inference call is as follows:

```
python predict_aac_scores.py --img_dir=<absolute path to folder containing DEXA images> --model_file_segmentation=<absolute path to segmentation model file> --model_file_regression=<absolute path to regression model file> --visualize=False
```

First, the images are segmented and the extracted Aortic regions are extracted to a sub-folder called 'aortic_regions' within the folder containing the DEXA images. Next, the regression model runs on these extracted regions to generate a csv file called 'predicted_aac_scores_model1.csv' in the main folder containing the DEXA images. A sample extract from the CSV file looks like the image below.

img_name | predicted_score
--- | --- 
4378704.png | 1.12
1996028.png | 3.15
4223931.png | 1.46
3078562.png | 1.92
4268644.png | 0.32

If ```--visualize=True```, then a segmentations image similar to the one shown below is saved for every image that is segmented and whose aortic region is extracted. In the image below, the subplots show - vertebrae segmentations + curve joining their centroids, pelvis segmentation + centroid, spinal curve + aortic region parallel to the curve, binary mask for the aortic region - from left to right in that order.

![ScreenShot](images/visualization_example.png)
