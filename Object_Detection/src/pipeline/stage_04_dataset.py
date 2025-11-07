

from src.components.dataset import MyDataset
from src.config.configuration import ConfigurationManager
from src.utils.logging_setup import logger

class DatasetPipeline:
    ''' 
    This class handles the dataset pipeline.
    '''
    def __init__(self, config: ConfigurationManager,):
        self.config = config.get_dataset_config()


    def run_pipeline(self, subset: str = None, transforms=None) -> MyDataset:
        '''
        Runs the dataset pipeline.

        Args:
            subset (str, optional): The subset of the dataset to load. Defaults to None.
            transforms (callable, optional): The transformation to apply to the dataset. Defaults to None.

        Returns:
            MyDataset: The dataset object.
        '''
        logger.info("Running dataset pipeline")
        dataset = MyDataset(self.config, subset, transforms)
        logger.info("Dataset pipeline completed")

        return dataset