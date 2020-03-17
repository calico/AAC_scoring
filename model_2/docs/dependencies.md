[Back to home.](../README.md)

# External Dependencies

This model has been upgraded to **run** in a `python 3` environment, cohesively with the independently-developed [Model 1](https://github.com/calico/AAC_scoring/tree/master/model_1).  A full list of dependencies for both pipelines can be found in the [requirements.txt](https://github.com/calico/AAC_scoring/blob/master/requirements.txt) file found in the [top-level repository](https://github.com/calico/AAC_scoring).

This model was **developed** in `Python 2.7.15` and was run in that environment for the manuscript by Sethi et al.  Here is a summary of the external dependencies directly used by [Model 2](https://github.com/calico/AAC_scoring/tree/master/model_2):

It uses the following standard Python modules for that version:
`os`, `argparse`, `math`, `sys`

It uses the following non-standard Python modules:
`numpy`, `scipy`

It uses `OpenCV`, called as `import cv2`.

It uses `tensorflow`.

It uses `keras`.  It also uses the following keras utility:
`from keras.models import load_model`.


