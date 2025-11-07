
from typing import Tuple
from src.components.data_transformation import MyTransform
from src.config.configuration import ConfigurationManager
from src.utils.logging_setup import logger

class DataTransformationPipeline:
    '''
    Data Transformation Pipeline'''
    def __init__(self, config: ConfigurationManager):
        """Initializes the Data Transformation Pipeline."""
        logger.info("Initializing data transformation pipeline")
        self.config = config

    def run_pipeline(self) -> Tuple[MyTransform, MyTransform]:
        """Runs the data transformation pipeline."""
        logger.info("Running data transformation pipeline")

        train_transforms = MyTransform(self.config.get_data_transformation_config(), train=True)
        valid_transforms = MyTransform(self.config.get_data_transformation_config(), train=False)
        logger.info("Data transformation pipeline completed")
        return train_transforms, valid_transforms
