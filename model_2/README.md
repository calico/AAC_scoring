# AAC_scoring

This repository includes the **second** (Model 2) pipeline for calculating
AAC scores from DEXA images used by Sethi et al (to-be-published).
[The top-level repository](https://github.com/calico/AAC_scoring) for the overall
project also contains
the [pipeline for the **first** (Model 1) method](https://github.com/calico/AAC_scoring/tree/master/model_1).
Both pipelines exclude
some images due to quality issues: Sethi et al only used images
for which **both** pipelines generated an AAC estimate for downstream
analysis (using the average of the two scores).  Both pipelines can be run from
a single script in the top-level repository.

## [AAC Analysis: Methods Description](docs/analysis.md)

## [Get Started: Running the Analysis](docs/getstarted.md)

## [External Dependencies](docs/dependencies.md)

## [Developer Documentation](docs/developer.md)
