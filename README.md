# Fusing Event-based and RGB camera for Robust Object Detection in Adverse Conditions
Authors: Abhishek Tomy, Anshul Paigwar, Khushdeep Singh Mann, Alessandro Renzaglia, Christian Laugier 

[Paper](https://hal.archives-ouvertes.fr/hal-03591717/) | [Project website](https://abhishek1411.github.io/event-rgb-fusion/) | [Video](https://www.youtube.com/watch?v=2nrqhiiXJwY&ab_channel=AbhishekTomy) | [Presentation](https://www.youtube.com/watch?v=xg3ExZV84Yg&ab_channel=AbhishekTomy)

This repository is official code release for the paper 'Fusing Event-based and RGB camera for Robust Object Detection in Adverse Conditions'. This research work was accepted in ICRA 2022 Philadelphia, USA.  The proposed approach uses Retinanet-50 architecture, with fusion of RGB and event features occuring at multiple levels in the feature pyramid net.

![object_detection1](https://user-images.githubusercontent.com/11161532/172838688-cab4a68e-cb83-4c00-88b5-95c758f5964a.gif)  

The main script to train the sensor fusion model is:

├── train_events.py

For running the code, the input is expected in the form of event voxels and RGB images. For creating event voxels, following codebase can be utilised:
[DSEC project webpage](https://github.com/uzh-rpg/DSEC)

# Abstract 
The ability to detect objects, under image corruptions and different weather conditions is vital for deep learning models especially when applied to real-world applications such as autonomous driving. Traditional RGB-based detection fails under these conditions and it is thus important to design a sensor suite that is redundant to failures of the primary frame-based detection. Event-based cameras can complement frame-based cameras in low-light conditions and high dynamic range scenarios that an autonomous vehicle can encounter during navigation. Accordingly, we propose a redundant sensor fusion model of event-based and frame-based cameras that is robust to common image corruptions. The method utilizes a voxel grid representation for events as input and proposes a two-parallel feature extractor network for frames and events. Our sensor fusion approach is more robust to corruptions by over 30% compared to only frame-based detections and outperforms the only event-based detection. The model is trained and evaluated on the publicly released DSEC dataset.

![Network_Architecture](https://user-images.githubusercontent.com/11161532/172815632-db193a8e-4c55-4572-aadc-87c22e6230a7.png)

# DSEC datset

The object detection labels where generated using [YOLOv5](https://github.com/ultralytics/yolov5) on the left RGB images and then transferred to the event frame using homographic transformation and refinement.

The expected structure of the data is given in the sample_data folder. The expected dataset follows the same structure as the DSEC dataset:
<pre>
├── DSEC  
    └── train   test  
        └── events                      images  
            └── interlaken_00_c         └── interlaken_00_c  
                ...                             ....
</pre>

# Homographic Transformation
The code for homographic transformation of RGB images to Event frame can be found here:
[homographic tansformation](https://github.com/RunqiuBao/fov_alignment/blob/main/fov_align.ipynb)

# Results
| Model        | Input           | mAP    | rPC   |
| :---         |    :----:       |   ---: | ---:  |
| RetinaNet-50 | Event Voxel     | 0.12   | -     |
| RetinaNet-50 | Event Gray      | 0.12   | -     |
| RetinaNet-50 | RGB             | 0.25   | 38.6  |
| Early Fusion | Event Gray+RGB  | 0.26   | 40.4  |
| FPN-Fusion   | Event Gray+RGB  | 0.25   | 60.8  |
| Early Fusion | Event Voxel+RGB | 0.24   | 66.2  |
| FPN-Fusion   | Event Voxel+RGB | 0.24   | 68.7  |
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
