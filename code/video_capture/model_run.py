import os
import argparse

import torch
import torch.nn as nn

from pytorch_i3d import InceptionI3d
import cv2
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)

args = parser.parse_args()

compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def run(init_lr=0.1,
#         max_steps=64e3,
#         mode='rgb',
#         batch_size=3 * 15,
#         save_model='',
#         weights=None, 
#         class_list=None,
#         frame_input=None):

#     if frame_input is None:
#         return

#     # setup the model
#     if mode == 'flow':
#         i3d = InceptionI3d(400, in_channels=2)
#         i3d.load_state_dict(torch.load('weights/flow_imagenet.pt', map_location=compute_device))
#     else:
#         i3d = InceptionI3d(400, in_channels=3)
#         i3d.load_state_dict(torch.loa('weights/rgb_imagenet.pt', map_location=compute_device))

#     i3d.replace_logits(num_classes)
#     i3d.load_state_dict(torch.load(weights, map_location=compute_device))  # Load model on CPU
#     i3d = nn.DataParallel(i3d)
#     i3d.to(compute_device)
#     i3d.eval()

#     predict_result = i3d(frame_input)
#     final_prediction_idx = torch.argmax(torch.mean(predict_result, dim=2)).item()

#     return class_list[final_prediction_idx]

# def run_on_tensor(weights, ip_tensor, num_classes):
#     i3d = InceptionI3d(400, in_channels=3)
#     # i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))

#     i3d.replace_logits(num_classes)
#     i3d.load_state_dict(torch.load(weights))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
#     i3d.cuda()
#     i3d = nn.DataParallel(i3d)
#     i3d.eval()

#     t = ip_tensor.shape[2]
#     ip_tensor.cuda()
#     per_frame_logits = i3d(ip_tensor)

#     predictions = F.upsample(per_frame_logits, t, mode='linear')

#     predictions = predictions.transpose(2, 1)
#     out_labels = np.argsort(predictions.cpu().detach().numpy()[0])

#     arr = predictions.cpu().detach().numpy()[0,:,0].T

#     plt.plot(range(len(arr)), F.softmax(torch.from_numpy(arr), dim=0).numpy())
#     plt.show()

#     return out_labels


# def get_slide_windows(frames, window_size, stride=1):
#     indices = torch.arange(0, frames.shape[0])
#     window_indices = indices.unfold(0, window_size, stride)

#     return frames[window_indices, :, :, :].transpose(1, 2)

def format_frame(frames) :
    edited_frames = []

    for frame in frames :
        w, h,c = frame.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            frame = cv2.resize(frame, dsize=(0, 0), fx=sc, fy=sc)
        frame = (frame / 255.) * 2 - 1

        edited_frames.append(frame)

    return np.asarray(edited_frames, dtype=np.float32)


class I3DModel:
    def __init__(self, weights_RGB=None, weight = None ,num_classes=100):
        if weights_RGB is None:
            print("Please provide weights for the model")
            return
        
        self.model = InceptionI3d(400, in_channels=3) #Always RGB Mode
        self.model.load_state_dict(torch.load(weights_RGB, map_location=compute_device))
        self.model.replace_logits(num_classes)

        self.model.load_state_dict(torch.load(weight, map_location=compute_device))
        self.model = nn.DataParallel(self.model)
        self.model.to(compute_device)
        self.model.eval()

    def run(self, frame_input):
        frame_input = format_frame(frame_input)
        predict_result = self.model(frame_input)
        final_prediction_idx = torch.argmax(torch.mean(predict_result, dim=2)).item()
        return final_prediction_idx


# if __name__ == '__main__':
#     # ================== test i3d on a dataset ==============
#     # need to add argparse
#     mode = 'rgb'
#     num_classes = 100
#     save_model = './checkpoints/'

#     weights = 'archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt' #Change the path whenever changing subset.

#     class_list = {}
#     with open('preprocess/wlasl_class_list.txt', 'r') as f:
#         for i in f :
#             index, value = i.strip().split("\t")
#             class_list[(int(index))] = value
    
#     run(mode=mode, save_model=save_model, weights=weights, class_list=class_list)
