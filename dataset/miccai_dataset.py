import os
from PIL import Image
import json
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset


class MICCAIDataset(Dataset):

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        img_file = self.images[index]['file_name']
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        caption = ann["caption"]
        instrument = ann['instrument']
        labels = ann['labels']
        short = ann['short']
        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "instrument": instrument,
            "labels": labels,
            "short": short,
        }

    def __init__(self, data_root):
        ann_path = os.path.join(data_root, 'annotations/train.json')
        self.vis_root=os.path.join(data_root, 'train')
        self.annotation = []
        self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])
        self.images = []
        self.images.extend(json.load(open(ann_path, "r"))['images'])
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            lambda x: x.half(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
