import torchvision.transforms.functional as F
from PIL import Image
import torch
import random
from typing import Tuple, Dict
from src.entity.config_entity import DataTransformationConfig


class MyTransform:
    """
    All-in-one transformation class for Object Detection.
    Handles sequential application of transforms (Compose logic), resizing,
    conversion to Tensor, and geometric augmentation (RandomHorizontalFlip) 
    while correctly updating bounding boxes.
    """
    def __init__(self, config: DataTransformationConfig, train: bool = False):
        """
        Args:
            config (DataTransformationConfig): Configuration for the Data Transformation Stage.
            train (bool, optional): Whether the model is in training mode. Defaults to False.
        """
        self.config = config
        self.train = train

    def _random_horizontal_flip(self, image: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        """Applies horizontal flip to image and updates bounding boxes."""
        if random.random() < self.config.flip_prob:
            image = F.hflip(image)
            
            # Get width (W) of the PIL image
            w, _ = image.size 
            
            # Clone boxes tensor to prevent in-place modification issues
            boxes = target["boxes"].clone()
            
            # Apply the horizontal flip transformation to coordinates:
            # new_xmin = W - old_xmax
            # new_xmax = W - old_xmin
            xmin_old = boxes[:, 0].clone()
            xmax_old = boxes[:, 2].clone()
            
            boxes[:, 0] = w - xmax_old
            boxes[:, 2] = w - xmin_old
            
            target["boxes"] = boxes
            
        return image, target
    
    def _resize(self, image: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        """Resizes the image and updates bounding boxes."""
        image = F.resize(image, self.config.image_size)
        
        # Get width (W) and height (H) of the resized PIL image
        w, h = image.size
        
        # Clone boxes tensor to prevent in-place modification issues
        boxes = target["boxes"].clone()
        
        # Apply the resize transformation to coordinates
        boxes[:, 0] = boxes[:, 0] * (w / self.config.image_size[0])
        boxes[:, 2] = boxes[:, 2] * (w / self.config.image_size[0])
        boxes[:, 1] = boxes[:, 1] * (h / self.config.image_size[1])
        boxes[:, 3] = boxes[:, 3] * (h / self.config.image_size[1])
        
        target["boxes"] = boxes

        return image, target

    def _to_tensor(self, image: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Converts the PIL image to a PyTorch Tensor."""
        # torchvision.transforms.functional.to_tensor converts PIL Image to float Tensor (0-1 range)
        return F.to_tensor(image), target

    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Executes the transformation pipeline.
        """
        # 0. Resize (Always mandatory)
        if self.config.resize:
            image, target = self._resize(image, target)
        
        # 1. Augmentation (Only applied during training)
        if self.train and self.config.flip_prob > 0.0:
            image, target = self._random_horizontal_flip(image, target)
        
        # 2. Conversion (Always mandatory)
        image, target = self._to_tensor(image, target)
            
        return image, target