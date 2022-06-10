### Fusing Event-based and RGB camera for Robust Object Detection in Adverse Conditions

The ability to detect objects, under image corruptions and different weather conditions is vital for deep learning models especially when applied to real-world applications such as autonomous driving. Traditional RGB-based detection fails under these conditions and it is thus important to design a sensor suite that is redundant to failures of the primary frame-based detection. Event-based cameras can complement frame-based cameras in low-light conditions and high dynamic range scenarios that an autonomous vehicle can encounter during navigation. Accordingly, we propose a redundant sensor fusion model of event-based and frame-based cameras that is robust to common image corruptions. The method utilizes a voxel grid representation for events as input and proposes a two-parallel feature extractor network for frames and events. Our sensor fusion approach is more robust to corruptions by over 30% compared to only frame-based detections and outperforms the only event-based detection. The model is trained and evaluated on the publicly released DSEC dataset.



## Datset and training
The DSEC dataset contains data from two event and frame-based cameras in stereo setup recorded from a moving vehicle in cities across Switzerland and data from both sensor modalities are publicly released. However, they lack The object detection labels which we generated using [YOLOv5](https://github.com/ultralytics/yolov5) on the left RGB images and then transferred to the event frame using homographic transformation and refinement.
![dsec](https://user-images.githubusercontent.com/11161532/173057592-c92be2c5-a915-48e3-bb9e-f352cafb8a07.png)

To evaluate robustness and efficacy of event-based camera to act as a redundant sensor modality, the RGB images were corrupted by adding noise, blur and weather conditions. The performance is evaluated over 5 severity levels for each of the 15 corruption types. The corruption method is adapted from this work by [D. Hendrycks et al](https://arxiv.org/pdf/1903.12261.pdf?ref=https://githubhelp.com). All the evaluated models are trained only clean dataset, only during testing, the RGB images are corrupted as shown in the figure. For sensor fusion models, image dropouts were used with a probaility of 0.15 during training.
  
  
![15_corruptImages](https://user-images.githubusercontent.com/11161532/173067756-206afeac-6129-48b6-b3df-8dc10cd1ff53.jpg)  

## Sensor Fusion Models
In this work, we propose a Feature Pyramid Net fusion (FPN) which uses ResNet-50 used as backbone for feature extraction as shown in figure below. Events and RGB features are fused at multiple scales before feeding it to the FPN layers.
![Network_Architecture](https://user-images.githubusercontent.com/11161532/172815632-db193a8e-4c55-4572-aadc-87c22e6230a7.png)

Apart from this, we also evaluate the efficacy of combining events and RGB channels early. In this work we have evaluated 2 type of event representations: 
-  a) [Events-Gray](https://arxiv.org/pdf/1906.07165.pdf) (converting events to a gray scale image) and 
-  b) [Event-voxels](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Unsupervised_Event-Based_Learning_of_Optical_Flow_Depth_and_Egomotion_CVPR_2019_paper.pdf)

## Results
| Model        | Input           | mAP    | rPC   |
| :---         |    :----:       |   ---: | ---:  |
| RetinaNet-50 | Event Voxel     | 0.12   | -     |
| RetinaNet-50 | Event Gray      | 0.12   | -     |
| RetinaNet-50 | RGB             | 0.25   | 38.6  |
| Early Fusion | Event Gray+RGB  | 0.26   | 40.4  |
| FPN-Fusion   | Event Gray+RGB  | 0.25   | 60.8  |
| Early Fusion | Event Voxel+RGB | 0.24   | 66.2  |
| FPN-Fusion   | Event Voxel+RGB | 0.24   | 68.7  |

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/abhishek1411/event-rgb-fusion/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
