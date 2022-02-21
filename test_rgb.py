import argparse
import collections

import numpy as np
import time
import math
import os

import torch
import torch.optim as optim
from torchvision import transforms
import pickle

from retinanet import model
from retinanet.dataloader_rgb import CSVDataset, collater, Resizer, AspectRatioBasedSampler, \
    Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

# from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', default='csv', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', default='/home/abhishek/connect/DSEC_detection_labels/events/labels_filtered_train.csv',
                        help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', default='/home/abhishek/connect/DSEC_detection_labels/events/labels_filtered_map.csv',
                        help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--root_img',default='/home/abhishek/connect/DSEC/train/transformed_images',help='dir to root rgb images in dsec format')
    # '/home/atomy/DSEC/train/train_images' '/home/atomy/corruptions/fog/severity_3'
    parser.add_argument('--root_event', default='/home/abhishek/connect/DSEC_events_img',help='dir to toot event files in dsec directory structure')
    parser.add_argument('--fusion', help='Type of fusion:1)early, fpn_fusion, multi-level', type=str, default='rgb')
    parser.add_argument('--checkpoint', help='location of pretrained file', default='./csv_rgb_homographic_retinanet_75.pt')
    parser.add_argument('--csv_test', default='/home/abhishek/connect/DSEC_detection_labels/events/labels_filtered_test_yolo_homographic.csv',
                        help='Path to file containing training annotations (see readme)')
    parser.add_argument('--eval_corruption', help='evaluate on the coorupted images', type=bool, default=True)
    parser.add_argument('--corruption_group', help='corruption group number', type=int, default=0)
    

    parser = parser.parse_args(args)
    dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,root_event_dir=parser.root_event,root_img_dir=parser.root_img,
                                         transform=transforms.Compose([Normalizer(), Resizer()]))
    dataloader_train = DataLoader(dataset_train, batch_size=8, num_workers=0, shuffle=True,collate_fn=collater)
    
    print('RGB homographic: change csv_test (yolo,yolo_homograhic) ,change the cooruption image folder and in test img folder(above),change the save-detect folder')


    # Create the model
    list_models = ['early_fusion', 'fpn_fusion', 'event', 'rgb']
    if parser.fusion in  list_models:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(),fusion_model=parser.fusion,pretrained=False)
    else:
        raise ValueError('Unsupported model fusion')

    use_gpu = True
    checkpoint = torch.load(parser.checkpoint)
    retinanet.load_state_dict(checkpoint['model_state_dict'])
    epoch_loss_all = checkpoint['loss']
    epoch_total = checkpoint['epoch']
    print(f'testing {parser.fusion} model')
    retinanet.eval()

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=100)

    retinanet.train()
    retinanet.module.freeze_bn()
    root_save_detect_folder = '/home/abhishek/save_detection'

    corruption_types = [['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur','glass_blur'],
                    ['motion_blur','zoom_blur', 'fog', 'snow','frost'],['brightness',
                    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']]

    # corruption_types = [['shot_noise']]

    corruption_list = corruption_types[parser.corruption_group]
    print(corruption_list)
    severity_list = [1,2,3,4,5]

    coco = True
    if parser.eval_corruption:
        for corruption in corruption_list:
            Average_precisions = {'person':[],'large_vehicle':[],'car':[]}
            start_c = time.time()
            for severity in severity_list:
                corruption_folder = f'/mnt/8tb-disk/DATASETS/DSEC/corruptions/{corruption}/severity_{severity}'
                save_detect_folder = os.path.join(root_save_detect_folder,parser.fusion,corruption,f'severity_{severity}')
                os.makedirs(save_detect_folder,exist_ok=True)  
                parser.root_img = corruption_folder
                dataset_val1 = CSVDataset(train_file= parser.csv_test, class_list=parser.csv_classes,
                                        root_event_dir=parser.root_event,root_img_dir=parser.root_img, transform=transforms.Compose([Normalizer(), Resizer()]))
                start = time.time()
                # print(f'{parser.fusion}, {corruption}, severity_{severity}')
                if coco:
                    mAP = csv_eval.evaluate_coco_map(dataset_val1, retinanet,save_detection = True,save_folder = save_detect_folder,
                                load_detection = False)
                    Average_precisions['person'].append(mAP[0])
                    Average_precisions['large_vehicle'].append(mAP[1])
                    Average_precisions['car'].append(mAP[2])


                else:
                    mAP = csv_eval.evaluate(dataset_val1, retinanet,save_detection = False,save_folder = save_detect_folder,
                                load_detection = True)
                    Average_precisions['person'].append(mAP[0][0])
                    Average_precisions['large_vehicle'].append(mAP[1][0])
                    Average_precisions['car'].append(mAP[2][0])
                    # print(f'time for severity: {time_since(start)}')
                    # print('#########################################')

            print(f'{parser.fusion}, {corruption}')


            for label_name in ['person','large_vehicle','car']:
                print('{}: {}'.format(label_name, list(np.around(np.mean(np.array(Average_precisions[label_name]),axis=1),2))))
                # print('{}: {}'.format(label_name, list(np.around(np.array(Average_precisions[label_name]),2))))
            print(f'time for corruption: {time_since(start_c)}')

            ap_file = os.path.join(save_detect_folder,f'{corruption}_ap.txt')
            with open(ap_file, "wb") as fp:
                pickle.dump(Average_precisions, fp)
    else:
        
        Average_precisions = {'person':[],'large_vehicle':[],'car':[]}
        dataset_val1 = CSVDataset(train_file= parser.csv_test, class_list=parser.csv_classes,
                                        root_event_dir=parser.root_event,root_img_dir=parser.root_img, transform=transforms.Compose([Normalizer(), Resizer()]))
        
        start = time.time()
        save_detect_folder = os.path.join(root_save_detect_folder,parser.fusion,'evaluation')
        # save_detect_folder = '/home/abhishek/save_detection/rgb_homographic/evaluation'
        os.makedirs(save_detect_folder,exist_ok=True)
        if coco:
            mAP = csv_eval.evaluate_coco_map(dataset_val1, retinanet,save_detection = True,save_folder = save_detect_folder,
                                load_detection = False)
            Average_precisions['person'].append(mAP[0])
            Average_precisions['large_vehicle'].append(mAP[1])
            Average_precisions['car'].append(mAP[2])


        else:
            mAP = csv_eval.evaluate(dataset_val1, retinanet,save_detection = True,save_folder = save_detect_folder,
                                load_detection = False)
            Average_precisions['person'].append(mAP[0][0])
            Average_precisions['large_vehicle'].append(mAP[1][0])
            Average_precisions['car'].append(mAP[2][0])

        for label_name in ['person','large_vehicle','car']:
                print('{}: {}'.format(label_name, list(np.mean(np.around(np.array(Average_precisions[label_name]),2),axis=1))))
                # print('{}: {}'.format(label_name, list(np.around(np.array(Average_precisions[label_name]),2))))

        ap_file = os.path.join(save_detect_folder,f'evaluation_ap.txt')
        with open(ap_file, "wb") as fp:
            pickle.dump(Average_precisions, fp)
        print(time_since(start))

if __name__ == '__main__':
    main()
