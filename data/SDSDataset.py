import os.path
from pathlib import Path
from typing import Union
import torch
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision.transforms import Compose, ToTensor, Resize

class SDSDataset(Dataset):
    CLASSES = ('ignored', 'swimmer', 'floater', 'boat')

    def __init__(
        self,
        root: Union[str, Path] = None,
        *,
        train=False,
    ):
        ann_file = os.path.join(root, 'truth.json')
        self.root = root
        self.train = train
        self.img_prefix = os.path.join(root, 'C3SM_spec')
        self.img_infos = self.load_annotations(ann_file)

        self.coco = COCO(ann_file)
        self.targets = [0 for _ in range(len(self.img_infos))]

        self.ids = [info['id'] for info in self.img_infos]
        self.num_classes = 4
        self.resize = [800, 1200]

        self.transform = Compose([
            Resize(self.resize),
            ToTensor()
            ])

    def __getitem__(self, index):
        coco = self.coco
        # Image ID of the input image
        img_id = self.ids[index]
        # Annotation IDs from coco
        ann_ids = coco.getAnnIds(img_id)
        # Load Annotation for the input image
        coco_annotation = coco.loadAnns(ann_ids)

        if self.train:
            coco_annotation = [ann for ann in coco_annotation if ann.get('iscrowd', 0) == 0]

        # Get path for the input image
        path = coco.loadImgs(img_id)[0]['file_name']
        org_image = Image.open(os.path.join(self.img_prefix, path))

        # Get size of input image
        org_height = org_image.height
        org_width = org_image.width

        # Apply transformation (resize) to input image
        image = self.transform(org_image)

        # Get number of objects in the input image
        num_objects = len(coco_annotation)

        boxes = []
        labels = []
        for i in range(num_objects):
            # Convert and resize boxes
            xmin = coco_annotation[i]['bbox'][0] / (org_width / self.resize[1])
            ymin = coco_annotation[i]['bbox'][1] / (org_height / self.resize[0])
            xmax = xmin + coco_annotation[i]['bbox'][2] / (org_width / self.resize[1])
            ymax = ymin + coco_annotation[i]['bbox'][3] / (org_height / self.resize[0])
            labels.append(coco_annotation[i]['category_id'])

            boxes.append([xmin, ymin, xmax, ymax])

        # Convert to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([img_id])

        # Get (rectangular) size of bbox
        areas = []
        for i in range(num_objects):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        # Get Iscrowd
        iscrowd = torch.zeros((num_objects,), dtype=torch.int64)

        # Create annotation dictionary
        annotation = dict()
        annotation['boxes'] = boxes
        annotation['labels'] = labels
        annotation['image_id'] = img_id
        annotation['area'] = areas
        annotation['iscrowd'] = iscrowd

        # Save width and height of the original image to rescale bounding boxes later on
        annotation['org_h'] = torch.as_tensor(org_height, dtype=torch.int64)
        annotation['org_w'] = torch.as_tensor(org_width, dtype=torch.int64)

        return image, annotation

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            ann_ids = self.coco.getAnnIds(i)
            coco_annotation = self.coco.loadAnns(ann_ids)
            coco_annotation = [ann for ann in coco_annotation if ann.get('iscrowd', 0) == 0]
            if len(coco_annotation) == 0:
                continue

            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)

        return img_infos

    def __len__(self):
        return len(self.img_infos)