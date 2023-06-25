
import Code.transforms as T
import os
import torch
import torch.utils.data

from torch.utils.data import random_split
from PIL import Image
from pycocotools.coco import COCO

TRAIN_DATA_DIR = '/home/locth/Documents/KLTN/urbandriving/images'
TRAIN_COCO = '/home/locth/Documents/KLTN/urbandriving/annotations/annotations.json'
NUM_WORKER = 0

# Batch size
BATCH_SIZE = 64

class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #Labels
        #labels = torch.ones((num_objs,), dtype=torch.int64)
        label = []
        for i in range(num_objs):
            category_id = coco_annotation[i]['category_id']
            label.append(category_id)
            labels = torch.as_tensor(label, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        convert = T.Compose([T.PILToTensor(),
                            T.ConvertImageDtype(torch.float)])
        img, my_annotation = convert(img, my_annotation)
        if self.transforms is not None:
            img, my_annotation = self.transforms(img, my_annotation)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


# transform function for both images and annotations
def get_transform(train):
    transforms = []
    #transforms.append(T.PILToTensor())
    #transforms.append(T.ConvertImageDtype(torch.uint8))
    if train:
        # transforms.append(T.RandomPhotometricDistort())
        transforms.append(T.RandomShortestSize(min_size = 320,max_size = 320))
        #transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def main_prepare_dataset():

    train_data_dir = TRAIN_DATA_DIR
    train_coco = TRAIN_COCO
    # create own Dataset
    my_dataset = myOwnDataset(root=train_data_dir,
                            annotation=train_coco,
                            transforms=get_transform(train = True)
                            )

    indices = torch.randperm(len(my_dataset)).tolist()
    print(len(my_dataset))
    train_size = int(0.5*len(my_dataset))
    val_size = int(0.3*len(my_dataset))
    test_size = len(my_dataset) - train_size - val_size
    # data_25 = len(my_dataset) - data_75
    print(train_size)
    print(val_size)
    print(test_size)

    # Created using indices from 0 to train_size.
    train_dataset = torch.utils.data.Subset(my_dataset, range(train_size))

    # Created using indices from train_size to train_size + test_size.
    # val_dataset = torch.utils.data.Subset(my_dataset, range(train_size, train_size + val_size))

    # # Created using indices from train_size to train_size + test_size.
    # test_dataset = torch.utils.data.Subset(my_dataset, range(train_size + val_size, train_size + val_size + test_size))

    # Set the random seed for reproducibility (optional)
    # torch.manual_seed(42)

    train_dataset_, val_dataset, test_dataset = random_split(my_dataset, [train_size, val_size, test_size])

    # own DataLoader
    data_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=NUM_WORKER,
                                            collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=NUM_WORKER,
                                            collate_fn=collate_fn)

    return data_loader, data_loader_test