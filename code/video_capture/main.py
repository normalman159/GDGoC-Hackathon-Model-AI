import os
import argparse
import cv2
import mediapipe as mp
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import videotransforms
import numpy as np
from pytorch_i3d import InceptionI3d

mp_hand = mp.solutions.hands
hands = mp_hand.Hands(static_image_mode=True, min_detection_confidence=0.5, max_num_hands = 2)

compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

i3d = InceptionI3d(400, in_channels=3)
i3d.load_state_dict(torch.load('rgb_imagenet.pt', map_location=compute_device))
i3d.replace_logits(100)
i3d.load_state_dict(torch.load('FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt', map_location=compute_device))  # Load model on CPU
i3d = torch.nn.DataParallel(i3d)
i3d.to(compute_device)
i3d.eval()


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


frames_1 = []
test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

def preprocess(frames) :
    edited_frames = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        w, h, c = frame.shape

        # Resize if smaller than 226x226
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            frame = cv2.resize(frame, dsize=(0, 0), fx=sc, fy=sc)

        frame = (frame / 255.) * 2 - 1  # Normalize to range [-1, 1]
        edited_frames.append(frame)
    return np.asarray(edited_frames, dtype=np.float32)

class Webcam(Dataset) :
    def __init__(self, transform=None, frames=None):
        self.frames = frames
        self.transform = transform
    
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):        
        frames = preprocess(self.frames)
        imgs = self.transform(frames)
        return video_to_tensor(imgs), None, None

def open_camera(frames_1=frames_1) :
    cap = cv2.VideoCapture(0)
    while True :
        ret, frame = cap.read()
        if ret is None: break
        frame = cv2.flip(frame, 1)

        hands_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
        if hands_results.multi_hand_landmarks:
            print("FIND HANDS!!!!!!!!!!!!!!!!!!!!!")
            frames_1.append(frame)
                
        elif len(frames_1) > 10:
            print("NO HANDS and detecting!!!!!!!!!!!", len(frames_1))

            val_dataset = Webcam(frames=frames_1, transform=test_transforms)
            val_dataloader, a,b = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

            per_frame_logits = i3d(val_dataloader)
            final_prediction_idx = torch.argmax(torch.mean(per_frame_logits, dim=2)).item()

            print(final_prediction_idx)

            frames_1 = []

        else:
            print("NO HANDS!!!!!!!!!!!!!!!!!!!!!", len(frames_1))
            frames_1 = []

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break         


def main() :
    open_camera(frames_1=frames_1)


if __name__ == '__main__':
    main()