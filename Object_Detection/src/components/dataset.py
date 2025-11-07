import torch
import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple
from src.entity.config_entity import DatasetConfig
from src.utils.logging_setup import logger

class MyDataset(Dataset):
    """
    Custom Dataset class for loading VOC-formatted data.
    Handles image loading and XML annotation parsing.
    """
    def __init__(self, config: DatasetConfig, subset: str = None, transforms=None) -> None:
        '''
        Initialize the dataset with configuration and optional transforms.
        '''
        self.config = config
        self.transforms = transforms

        subset_dir = None
        if subset == "train":
            subset_dir = self.config.train_dir 
        elif subset == "valid":
            subset_dir = self.config.valid_dir
        elif subset == "test":
            subset_dir = self.config.test_dir
        else:
            raise ValueError(f"Invalid subset: {subset}. Must be one of 'train', 'valid', or 'test'.")
        
        self.root_dir = os.path.join(self.config.root_dir, subset_dir)

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset root directory {self.root_dir} does not exist.")
        
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"Dataset root directory {self.root_dir} is not a directory.")
        
        if not os.listdir(self.root_dir):
            raise ValueError(f"Dataset root directory {self.root_dir} is empty.")
        

        # Collect base IDs for image/annotation pairs
        self.ids = []
        files = os.listdir(self.config.root_dir)
        
        # Collect unique base names by looking for XML files
        xml_files = [f for f in files if f.endswith(".xml")]
        
        for xml_file in xml_files:
            # Base ID includes the `_jpg.rf.HASH` part as it's common to both files
            base_id = os.path.splitext(xml_file)[0] 
            img_path_check = os.path.join(self.root_dir, base_id + '.jpg')
            
            if os.path.exists(img_path_check):
                self.ids.append(base_id)
        
        if not self.ids:
            logger.warning(f"Warning: No matching image/annotation pairs found in {self.config.root_dir}")

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.ids)

    def _parse_xml(self, ann_path: str, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parses a VOC XML file to extract bounding boxes and labels."""
        boxes = []
        labels = []
        try:
            # Parse the XML file
            logger.info(f"Parsing XML for {ann_path}")
            tree = ET.parse(ann_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                label_name = obj.find('name').text
                if label_name in self.config.class_map:
                    label = self.config.class_map[label_name]
                    
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(label)
        except Exception as e:
            logger.error(f"Error parsing XML for {ann_path}: {e}")
            
        # Handle the case where an image has NO annotations
        if not boxes:
            boxes.append([0, 0, 1, 1])
            labels.append(0)  # Background label

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        logger.info(f"Image {idx} has {len(boxes)} objects")
        return boxes, labels

    def __getitem__(self, idx: int):
        """Returns a sample from the dataset at the given index."""
        if idx >= len(self.ids):
            raise IndexError(f"Index {idx} is out of range. Dataset has {len(self.ids)} samples.")
        logger.info(f"Loading sample {idx}")
        base_id = self.ids[idx]
        img_path = os.path.join(self.root_dir, f"{base_id}.jpg")
        ann_path = os.path.join(self.root_dir, f"{base_id}.xml")

        img = Image.open(img_path).convert("RGB")
        boxes, labels = self._parse_xml(ann_path, idx)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        logger.info(f"Sample {idx} loaded with {len(boxes)} objects")
        # Apply transforms
        if self.transforms:
            logger.info(f"Applying transforms to sample {idx}")
            img, target = self.transforms(img, target)

        logger.info(f"Sample {idx} transformed")
        return img, target

