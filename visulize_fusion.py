import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
from retinanet import model
from retinanet.dataloader import CSVDataset_event, collater, Resizer, AspectRatioBasedSampler, \
    Augmenter, \
    Normalizer
from torchvision import transforms
from torch.utils.data import DataLoader

def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption,colour):
    b = np.array(box).astype(int)
    # cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    # cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, colour, 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, colour, 1)


def Normalizer(sample):
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])

    return ((sample.astype(np.float32)-mean)/std)


def detect_image(image_path, model_path, class_list):

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key


    for img_name in os.listdir(image_path):

        event_file = os.path.join(image_path,img_name)
        image = torch.from_numpy(np.load(event_file)['arr_0'])

        file = image_path.split('/')
        img_file = os.path.join(parser.root_img, file[-2], 'images/left/rectified', img_name.replace('.npz', '.png'))
        img_rgb = cv2.imread(img_file)

        ev_img = torch.sum(image, axis=0).numpy()
        # ev_img = (ev_img / ev_img.max() * 256).astype('uint8')
        left_event_b = ev_img * 127 / abs(ev_img).max() + 127
        ret, left_event_b = cv2.threshold(left_event_b, 126, 255, cv2.THRESH_BINARY)
        left_event_3c = np.repeat(left_event_b[:, :, np.newaxis], 3, axis=2)

        combined = cv2.addWeighted(img_rgb[:, :, [2, 1, 0]].astype(np.float64), 0.7, (left_event_3c).astype(np.float64),0.3, 0)
        # cv2.imwrite(os.path.join(save_path, os.path.splitext(img_name)[0] + '.png'), combined)
        # combined = img_rgb

        img_rgb = img_rgb.astype(np.float32) / 255.0
        img_rgb = Normalizer(img_rgb)
        image = np.expand_dims(image, 0)
        img_rgb = np.expand_dims(img_rgb,0)
        img_rgb = np.transpose(img_rgb, (0, 3, 1, 2))

        with torch.no_grad():

            image = torch.from_numpy(image)
            img_rgb = torch.from_numpy(img_rgb)
            if torch.cuda.is_available():
                image = image.cuda().float()
                img_rgb = img_rgb.cuda().float()

            st = time.time()

            scores, classification, transformed_anchors = retinanet((img_rgb, image))
            # print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)
            # ev_img = cv2.cvtColor(ev_img, cv2.COLOR_GRAY2RGB)
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = labels[int(classification[idxs[0][j]])]
                # print(bbox, classification.shape)
                score = scores[idxs[0][j]]
                caption = '{} {:.3f}'.format(label_name, score)
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                draw_caption(combined, (x1, y1, x2, y2), caption,(0, 0, 255))
                cv2.rectangle(combined, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)

            # cv2.imshow('detections', ev_img)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(save_path,os.path.splitext(img_name)[0]+'.png'), combined)
            # print('Done')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', default= '/media/storage/DSEC_events_img/zurich_city_01_e/left',help='Path to directory containing images')
    parser.add_argument('--class_list', default='/media/storage/DSEC_detection_labels/events/labels_filtered_map.csv',help='Path to CSV file listing class names (see README)')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--fusion', help='Type of fusion:1)early, fpn_fusion, multi-level', type=str,default='fpn_fusion')
    parser.add_argument('--csv_train', default='/media/storage/DSEC_detection_labels/events/labels_filtered_train.csv',help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', default='/media/storage/DSEC_detection_labels/events/labels_filtered_map.csv', help='Path to file containing class list (see readme)')
    parser.add_argument('--root_img', default='/home/Abhishek/Desktop/connect_8tb/DSEC/corruptions/brightness/severity_5', help='dir to toot rgb images in dsec format')
    # / media / storage / DSEC / train / transformed_images
    parser.add_argument('--root_event', default='/media/storage/DSEC_events_img', help='dir to toot event files in dsec directory structure')
    parser.add_argument('--model_path', default='./csv_fpn_homographic_retinanet_75.pt', help='Path to model')

    parser = parser.parse_args()

    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=3, pretrained=False)
        # retinanet = torch.load('csv_retinanet_1.pt')
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=3,fusion_model=parser.fusion, pretrained=False)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    checkpoint = torch.load(parser.model_path)
    retinanet.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()


    retinanet.training = False
    retinanet.eval()
    save_path = '/media/storage/event_images/zurich_city_01_e_predictions_fusion_brightness5'
    os.makedirs(save_path, exist_ok=True)
    detect_image(parser.image_dir, parser.model_path, parser.class_list)
