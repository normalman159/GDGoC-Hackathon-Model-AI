import os
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import transforms
import videotransforms

import numpy as np

import torch.nn.functional as F
from pytorch_i3d import InceptionI3d

from datasets.nslt_dataset_all import NSLT as Dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)

args = parser.parse_args()

compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(init_lr=0.1,
        max_steps=64e3,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        batch_size=3 * 15,
        save_model='',
        weights=None, class_list=None):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)

    for i, data in enumerate(val_dataset) :
        if i >= 1:
            break
        print(data[0].shape)
        

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=2,
                                                 pin_memory=False)
    for i, batch in enumerate(val_dataloader):
        if i >= 2:
            break
        print(batch[0].shape)
        print(len(batch[0]))
    dataloaders = {'test': val_dataloader}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('weights/flow_imagenet.pt', map_location=compute_device))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt', map_location=compute_device))

    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights, map_location=compute_device))  # Load model on CPU
    i3d = nn.DataParallel(i3d)
    i3d.to(compute_device)
    i3d.eval()

    

    # for data in dataloaders["test"]:
    #     inputs, labels, video_id = data  # inputs: b, c, t, h, w
    #     if inputs == None : continue
    #     per_frame_logits = i3d(inputs)

    #     final_prediction_idx = torch.argmax(torch.mean(per_frame_logits, dim=2)).item()

    #     print(video_id ,class_list[labels[0].item()], class_list[final_prediction_idx])


def run_on_tensor(weights, ip_tensor, num_classes):
    i3d = InceptionI3d(400, in_channels=3)
    # i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))

    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    t = ip_tensor.shape[2]
    ip_tensor.cuda()
    per_frame_logits = i3d(ip_tensor)

    predictions = F.upsample(per_frame_logits, t, mode='linear')

    predictions = predictions.transpose(2, 1)
    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])

    arr = predictions.cpu().detach().numpy()[0,:,0].T

    plt.plot(range(len(arr)), F.softmax(torch.from_numpy(arr), dim=0).numpy())
    plt.show()

    return out_labels


def get_slide_windows(frames, window_size, stride=1):
    indices = torch.arange(0, frames.shape[0])
    window_indices = indices.unfold(0, window_size, stride)

    return frames[window_indices, :, :, :].transpose(1, 2)


if __name__ == '__main__':
    # ================== test i3d on a dataset ==============
    # need to add argparse
    mode = 'rgb'
    num_classes = 100
    save_model = './checkpoints/'

    root = '../../data/WLASL{}'.format(num_classes)
    train_split = 'preprocess/nslt_{}.json'.format(num_classes)
    weights = 'archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt' #Change the path whenever changing subset.

    class_list = {}
    with open('preprocess/wlasl_class_list.txt', 'r') as f:
        for i in f :
            index, value = i.strip().split("\t")
            class_list[(int(index))] = value
    
    run(mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights, class_list=class_list)
