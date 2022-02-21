import argparse
import collections

import numpy as np
import time
import math

import torch
import torch.optim as optim
from torchvision import transforms

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
    base_dir = '/home/abhishek/connect'
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', default='csv', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', default=f'{base_dir }/DSEC_detection_labels/events/labels_filtered_train_yolo.csv',
                        help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', default=f'{base_dir }/DSEC_detection_labels/events/labels_filtered_map.csv',
                        help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--root_img',default=f'{base_dir }/DSEC/train/train_images', help='dir to root rgb images in dsec format')
    # '/home/atomy/DSEC/train/train_images' '/home/atomy/corruptions/fog/severity_3'
    parser.add_argument('--root_event', default=f'{base_dir }/DSEC_events_img',
                        help='dir to toot event files in dsec directory structure')
    parser.add_argument('--fusion', help='Type of fusion:1)early, fpn_fusion, multi-level', type=str, default='rgb')
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=10)
    parser.add_argument('--continue_training', help='load a pretrained file', default=True)
    parser.add_argument('--checkpoint', help='location of pretrained file', default='./csv_rgb_orig_retinanet_65.pt')


    parser = parser.parse_args(args)

    if parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,root_event_dir=parser.root_event,root_img_dir=parser.root_img,
                                         transform=transforms.Compose([Normalizer(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    # sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, batch_size=8, num_workers=6, shuffle=True,collate_fn=collater)
    dataset_val1 = CSVDataset(train_file=f'{base_dir }/DSEC_detection_labels/events/labels_filtered_test_yolo.csv', class_list=parser.csv_classes,
                                    root_event_dir=parser.root_event,root_img_dir=parser.root_img, transform=transforms.Compose([Normalizer(), Resizer()]))
    dataloader_val1 = DataLoader(dataset_val1 , batch_size=1, num_workers=6, shuffle=True,collate_fn=collater)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    list_models = ['fpn_fusion', 'event', 'rgb']
    if parser.fusion in  list_models:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(),fusion_model=parser.fusion,pretrained=False)
    else:
        raise ValueError('Unsupported model fusion')

    use_gpu = True
    if parser.continue_training:
        checkpoint = torch.load(parser.checkpoint)
        retinanet.load_state_dict(checkpoint['model_state_dict'])
        epoch_loss_all = checkpoint['loss']
        epoch_total = checkpoint['epoch']
        # epoch_loss_all = []
        print('training rgb model')
    else:
        epoch_total = 0
        epoch_loss_all =[]
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

    print('Num training images: {}'.format(len(dataset_train)))
    num_batches = 0
    start = time.time()
    print('RGB, impulse_noise images')
    # mAP = csv_eval.evaluate(dataset_val1, retinanet)
    print(time_since(start))
    epoch_loss = []
    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_total += 1 
        for iter_num, data in enumerate(dataloader_train):
            try:
                classification_loss, regression_loss = retinanet([data['img_rgb'],data['img'].cuda().float(),data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                num_batches += 1
                if num_batches == 8:  # optimize every 5 mini-batches
                    optimizer.step()
                    optimizer.zero_grad()
                    num_batches = 0


                loss_hist.append(float(loss))

                
                if iter_num % 500 ==0:
                    print(
                        '[rgb] [{}], Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            time_since(start), epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                    epoch_loss.append(np.mean(loss_hist))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        # torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))
        if epoch_num % 5 ==0:
            torch.save({'epoch': epoch_total, 'model_state_dict': retinanet.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': np.append(epoch_loss_all,epoch_loss)}, f'{parser.dataset}_rgb_dropout_retinanet_{epoch_total}.pt')

    # retinanet.eval()

    torch.save({'epoch': epoch_total, 'model_state_dict': retinanet.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': np.append(epoch_loss_all,epoch_loss)}, f'{parser.dataset}_rgb_dropout_retinanet_{epoch_total}.pt')


if __name__ == '__main__':
    main()
