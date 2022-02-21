import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
from retinanet import model

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


def detect_image(image_path, model_path, class_list):

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key



    for img_name in os.listdir(image_path):

        image = torch.load(os.path.join(image_path,img_name))
        ev_img = torch.sum(image, axis=0).numpy()
        ev_img = (ev_img/ ev_img.max() * 256).astype('uint8')

        image = np.expand_dims(image, 0)
        # image = np.transpose(image, (0, 2, 3, 1))

        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            print(image.shape)
            scores, classification, transformed_anchors = retinanet(image.cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)
            ev_img = cv2.cvtColor(ev_img, cv2.COLOR_GRAY2RGB)
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = labels[int(classification[idxs[0][j]])]
                print(bbox, classification.shape)
                score = scores[j]
                caption = '{} {:.3f}'.format(label_name, score)
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                draw_caption(ev_img, (x1, y1, x2, y2), caption,(0, 0, 255))
                cv2.rectangle(ev_img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)

            # cv2.imshow('detections', ev_img)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(save_path,os.path.splitext(img_name)[0]+'.png'), ev_img)
            print('Done')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', default= '/media/Abhishek/My Passport/DSEC_events_img/zurich_city_05_b/left',help='Path to directory containing images')
    parser.add_argument('--model_path', default='/home/Abhishek/Deep_networks/pytorch-retinanet/csv_retinanet_16.pt',help='Path to model')
    parser.add_argument('--class_list', default='/media/storage/DSEC_detection_labels/events/labels_filtered_map.csv',help='Path to CSV file listing class names (see README)')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser = parser.parse_args()

    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=6, pretrained=False)
        # retinanet = torch.load('csv_retinanet_1.pt')
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=6, pretrained=False)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    checkpoint = torch.load(parser.model_path)
    retinanet.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()

    retinanet.training = False
    retinanet.eval()
    save_path = '/media/storage/event_images/zurich_5b_predictions'
    detect_image(parser.image_dir, parser.model_path, parser.class_list)
