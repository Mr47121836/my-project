import torch
from torch.utils.data import Dataset
from torchvision import datasets

class MyDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(MyDataset, self).__init__()
        self.annotations_line = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __getitem__(self, item):

        return 1,1,3

    def __len__(self):
        return self.annotations_line