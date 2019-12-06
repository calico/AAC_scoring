[Back to home.](../README.md)

# Get Started: Running the Analysis

These python scripts use locally-referenced modules, so they should run in-place
from a clone of this repo.  However, they do rely on several external python
dependencies, [described here](dependencies.md).  There are two sections below:
**Basic execution** describes running the entire pipeline from a single command-line
call.  **Step-by-step execution** describes how to run the [three steps of analysis](analysis.md)
individually.

## Basic execution

End-to-end analysis can be performed using the `predict_aac_scores_2.py` script:

```shell
# end-to-end, image-directory-to-score-file execution
python predict_aac_scores_2.py -i ${IMAGE_DIR} -s ${STORAGE_DIR} -o ${RESULT_FILE} [--original_resize/--no_resize]
```

Required arguments:
- `IMAGE_DIR`: a directory of source images.  The file names, minus appends, will be
  used as the identifiers in the output file.  Corrupted or non-image files and
  sub-directories will be skipped over.
- `STORAGE_DIR`: intermediate processing files will be stored in this directory.
  They will remain after execution.  Re-calling this script will re-generate all
  intermediate files rather than pick up from where a prior incomplete run left off.
  To execute pipeline steps individually, on specific indermediate data, see the
  step-by-step instructions below.
- `RESULT_FILE`: a two-column, tab-delimited file with DEXA photo ID's in the left
  column and predicted AAC scores in the right.

**Optional arguments** dealing with the resizing of images (see [methods](analysis.md)
documentation for a description of these behaviors; **default** is to re-size all
images to give them a height of 940 pixels):
- `--original_resize`: specifies that images used by Sethi et al will be re-sized
  to the dimensions used in that study.  Unrecognized files will be re-sized by
  the default behavior (to a height of 940 pixels).  REQUIRES that image files
  from Sethi et al be named `<ID>.png`, where `ID` is the anonymized numeric ID
  from UK BioBank.
- `--no_resize`: specifies that no resizing should be performed on any images.


## Step-by-step execution

Each of the three analytical steps described in [AAC Analysis](analysis.md) can be
executed separately.  This pipeline is not engineered to be time-efficient, so for
large data sets (~1000 images or more), if you have access to a cluster, I
suggest parallelizing at least the first step and possibly the second step as well.
This can be easily achieved by splitting images up into separate directories; no
tools are provided for executing analysis on a subset of images in a directory.

### Step 1: isolation of the aorta via labelling of the spine.

```shell
# isolation of aortic images from full DEXA scans
python runMakeAortaPics.py -i ${INPUT_DIR} -o ${OUTPUT_DIR} [--original_resize/--no_resize]
```

Required arguments:
- `INPUT_DIR`: a directory of source images.  The equivalent of `INPUT_DIR` from
  `predict_aac_scores_2.py`.
- `OUTPUT_DIR`: the directory into which aorta-only images will be written.
  This is the equivalent of `STORAGE_DIR` from `predict_aac_scores_2.py`.
  It will also serve as the `INPUT_DIR` for `runApplyClassifiers.py` below.

**Optional arguments** deal with the resizing of images, and are the same as for
  `predict_aac_scores_2.py`.  They are described above, under **Basic execution**.
- `--original_resize`: see above.
- `--no_resize`: see above.

### Step 2: classification of calcification and noise in aorta images.

```shell
# the running of classification models on aortic images
python runApplyClassifiers.py -i ${INPUT_DIR} -o ${OUTPUT_FILE}
```

Required arguments:
- `INPUT_DIR`: a directory of aorta images.  The equivalent of `OUTPUT_DIR` from
  `runMakeAortaPics.py`.  Also equivalent to `STORAGE_DIR` from
  `predict_aac_scores_2.py`.
- `OUTPUT_FILE`: the base filename for tab-delimited (`.tsv`) files that will be written
  with the output of various models.  One file will be written for each model that is
  run.  For the arg `-o foo.tsv`, a file of low-threshold calcification calls ("calc" or
  "nocalc") will be created called `foo.aacLo.tsv`; a file of high-threshold calcification
  calls ("calc" or "nocalc") will be created called `foo.aacHi.tsv`; and a file of
  calls of the degree of background noise in the image ("low", "medium", or "high") will
  be created called `foo.back.tsv`.  The master script `predict_aac_scores_2.py` will
  place these files in that script's specified `STORAGE_DIR` directory, using the base filename `results.tsv`.
  The three files `foo.aacLo.tsv`, `foo.aacHi.tsv`, and `foo.back.tsv` will appropriately
  serve as the inputs for the args `--data_low`, `--data_high`, and `--data_bg` for the next
  script, `runFinalRegression.py`, described below.

### Step 3: scoring of aortic calcification using classification results.

```shell
# the use of classification scores to numerically estimate AAC
python runFinalRegression.py --data_low ${DF1} --data_high ${DF2} --data_bg ${DF3} -o ${OUTPUT_FILE}
```

Required arguments:
- `DF1`: a tab-delimited (.tsv) file of low-threshold AAC classifications and scores.
  The equivalent of the `foo.aacLo.tsv` output file from `runApplyClassifiers.py`.
- `DF2`: a tab-delimited (.tsv) file of high-threshold AAC classifications and scores.
  The equivalent of the `foo.aacHi.tsv` output file from `runApplyClassifiers.py`.
- `DF3`: a tab-delimited (.tsv) file of background-noise classifications and scores.
  The equivalent of the `foo.back.tsv` output file from `runApplyClassifiers.py`.
- `OUTPUT_FILE`: the name of the results file that will be written: a two-column,
  tab-delimited file with DEXA photo ID's in the left column and predicted AAC
  scores in the right.  This is the equivalen of `RESULT_FILE` from `predict_aac_scores_2.py`.


