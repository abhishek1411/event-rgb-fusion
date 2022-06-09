## Fusing Event-based and RGB camera for Robust Object Detection in Adverse Conditions

The ability to detect objects, under image corruptions and different weather conditions is vital for deep learning models especially when applied to real-world applications such as autonomous driving. Traditional RGB-based detection fails under these conditions and it is thus important to design a sensor suite that is redundant to failures of the primary frame-based detection. Event-based cameras can complement frame-based cameras in low-light conditions and high dynamic range scenarios that an autonomous vehicle can encounter during navigation. Accordingly, we propose a redundant sensor fusion model of event-based and frame-based cameras that is robust to common image corruptions. The method utilizes a voxel grid representation for events as input and proposes a two-parallel feature extractor network for frames and events. Our sensor fusion approach is more robust to corruptions by over 30% compared to only frame-based detections and outperforms the only event-based detection. The model is trained and evaluated on the publicly released DSEC dataset.



Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

# DSEC datset
The object detection labels where generated using [YOLOv5](https://github.com/ultralytics/yolov5) on the left RGB images and then transferred to the event frame using homographic transformation and refinement.

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
