
from Object_Detection.src.components.dataset import MyDataset
from src.components.data_loader import MyDataloader
from src.config.configuration import ConfigurationManager
from src.utils.logging_setup import logger

class DataLoaderPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.data_loader_config = self.config.get_data_loader_config()

    def run_pipeline(self, dataset: MyDataset = None, shuffle: bool = True):
       '''
       Runs the data loader pipeline.
       '''
       logger.info("Running data loader pipeline")
       data_loader = MyDataloader(config=self.data_loader_config, dataset=dataset, shuffle=shuffle)
       logger.info("Data loader pipeline completed")

       return data_loader