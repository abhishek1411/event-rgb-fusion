# Introduction
This the project folder for the paper `Fusing Event-based and RGB camera for Robust Object Detection in Adverse Conditions'. The proposed approach uses Retinanet-50 architecture, with fusion of RGB and event features occuring at multiple levels in the feature pyramid net.

The main script to train the sensor fusion model is:

├── train_events.py

For running the code, the input is expected in the form of event voxels and RGB images. For creating event voxels, following codebase can be utilised:
[project webpage](https://github.com/uzh-rpg/DSEC)


# DSEC datset

The object detection labels where generated using [YOLOv5](https://github.com/ultralytics/yolov5) on the left RGB images and then transferred to the event frame using homographic transformation and refinement.

The expected structure of the data is given in the sample_data folder. The expected dataset follows the same structure as the DSEC dataset:
├── DSEC
    └── train   test
        └── events                      images
            └── interlaken_00_c            └── interlaken_00_c
                ...                             ....
## Install

Install conda environment to run example code
```bash
conda create -n dsec python=3.8
conda activate dsec
conda install -y -c anaconda numpy
conda install -c anaconda pandas
conda install -c conda-forge pycocotools
conda install -c conda-forge opencv
conda install -c anaconda requests
conda install -y -c pytorch pytorch torchvision cudatoolkit=10.2
conda install -y -c conda-forge matplotlib
```

## Acknowledgements
The retinanet based sensor fusion model presented here builds upon this [implementation](https://github.com/yhenon/pytorch-retinanet)
