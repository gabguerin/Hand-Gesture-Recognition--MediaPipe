import csv
import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from HandModel import HandModel


class HandGestureDataset(Dataset):
    """Loads pose samples from a given folder.

    Required folder structure:
      hand_gesture_1.csv
      hand_gesture_2.csv
      ...

    Required CSV structure:
      sample_0001,x1,y1,z1,x2,y2,z2,....
      sample_0002,x1,y1,z1,x2,y2,z2,....
      ...
    """

    def __init__(self, dataset_folder="dataset"):
        # Each file in the folder represents one hand gesture class.
        file_names = [name for name in os.listdir(dataset_folder) if name.endswith(".csv")]
        self.classes = []
        self.class_to_idx = {}

        hand_gestures, labels = [], []
        for file_name in file_names:
            # Use file name as label
            klass = file_name[: -len(".csv")]
            if klass not in self.class_to_idx:
                label = len(self.classes)
                self.classes.append(label)
                self.class_to_idx[klass] = label
            else:
                label = self.class_to_idx[klass]

            # Parse CSV
            with open(os.path.join(dataset_folder, file_name)) as csv_file:
                csv_reader = list(csv.reader(csv_file, delimiter=","))
                for row in csv_reader[1:]:

                    landmarks = np.array(row[2:], np.float32).reshape([21, 3])

                    hand_gesture = HandModel(landmarks)
                    hand_gestures.append(hand_gesture.landmarks)
                    labels.append(label)

        self.hand_gestures = hand_gestures
        self.labels = labels

    def __len__(self):
        return len(self.hand_gestures)

    def __getitem__(self, idx):
        return {"landmark": self.hand_gestures[idx], "class": self.labels[idx]}
