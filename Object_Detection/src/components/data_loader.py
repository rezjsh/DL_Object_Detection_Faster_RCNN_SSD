from src.components.dataset import MyDataset
from src.entity.config_entity import DataLoaderConfig
from src.utils.helpers import collate_fn
from torch.utils.data import DataLoader
from src.utils.logging_setup import logger
class MyDataloader:
    '''
    Custom DataLoader for object detection.
    '''
    def __init__(self, config: DataLoaderConfig, dataset: MyDataset, shuffle: bool = True):
        self.config = config
        self.dataset = dataset
        self.shuffle = shuffle
        self.collate_fn = collate_fn

    def get_loader(self):
        '''
        Returns the DataLoader object.
        '''
        logger.info(f"Initializing DataLoader with batch size {self.config.train_batch_size if self.shuffle else self.config.valid_batch_size}, workers {self.config.num_workers}, shuffle={self.shuffle}.")
        data_loader = DataLoader(
            self.dataset,
            batch_size=self.config.train_batch_size if self.shuffle else self.config.valid_batch_size,
            shuffle=self.shuffle,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn
        )
        logger.info("DataLoader initialized.")
        return data_loader

    def __len__(self):
        '''
        Returns the number of batches in the DataLoader.
        '''
        return len(self.dataset) // (self.config.train_batch_size if self.shuffle else self.config.valid_batch_size)
    