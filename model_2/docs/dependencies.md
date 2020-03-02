[Back to home.](../README.md)

# External Dependencies

This system was developed for `Python 2.7.15`.

It uses the following standard Python modules for that version:
`os`, `argparse`, `math`, `sys`

It uses the following non-standard Python modules:
`numpy`, `scipy`

It uses `OpenCV`, called as `import cv2`.

It uses `tensorflow`.  It also uses the following tensorflow utility:
`from utils import label_map_util`.  This utility is part of the tensorflow research models package (installation instructions [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)).

It uses `keras`.  It also uses the following keras utility:
`from keras.models import load_model`.


Specific versions are listed in [requirements.txt](../requirements.txt).

