import os
from pathlib import Path
from src.entity.config_entity import DataIngestionConfig, DataLoaderConfig, DataTransformationConfig, DataValidationConfig, DatasetConfig
from src.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.utils.helpers import create_directory, read_yaml_file
from src.utils.logging_setup import logger
from src.core.singleton import SingletonMeta

class ConfigurationManager(metaclass=SingletonMeta):
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH, params_file_path: str = PARAMS_FILE_PATH):
        self.config = read_yaml_file(config_file_path)
        self.params = read_yaml_file(params_file_path)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        logger.info("Getting data ingestion config")
        config = self.config.data_ingestion
        params = self.params.data_ingestion
        logger.info(f"Data ingestion config: {config}")
        logger.info(f"Data ingestion params: {params}")

        dirs_to_create = [config.download_location]
        logger.info(f"Dirs to create: {dirs_to_create}")
        create_directory(dirs_to_create)
        logger.info("Creating data ingestion config")

        data_ingestion_config = DataIngestionConfig(
            download_location=config.download_location,
            workspace=params.workspace,
            project_name=params.project_name,
            version=params.version,
            format=params.format
        )
        logger.info(f"Data ingestion config created: {data_ingestion_config}")
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        logger.info("Getting data validation config")
        config = self.config.data_validation
        params = self.params.data_validation
        logger.info(f"Data validation config: {config}")
        logger.info(f"Data validation params: {params}")

        dirs_to_create = [os.path.dirname(config.status_file)]
        logger.info(f"Dirs to create: {dirs_to_create}")
        create_directory(dirs_to_create)
        logger.info("Creating data validation config")

        data_validation_config = DataValidationConfig(
            data_dir=config.data_dir,
            required_files=params.required_files,
            status_file=Path(config.status_file)
        )
        logger.info(f"Data validation config created: {data_validation_config}")
        return data_validation_config
    
    def get_dataset_config(self) -> DatasetConfig:
        logger.info("Getting dataset config")
        config = self.config.dataset
        logger.info(f"Dataset config: {config}")
        dataset_config = DatasetConfig(
            data_dir=Path(config.data_dir),
            train_dir=Path(config.train_dir),
            valid_dir=Path(config.valid_dir)
        )
        logger.info(f"Dataset config created: {dataset_config}")
        return dataset_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        logger.info("Getting data transformation config")
        params = self.params.data_transformation
        logger.info(f"Data transformation config: {params}")
        data_transformation_config = DataTransformationConfig(
            resize=params.resize,
            image_size=params.image_size,
            flip_prob=params.flip_prob
        )
        logger.info(f"Data transformation config created: {data_transformation_config}")
        return data_transformation_config
    
    def get_data_loader_config(self) -> DataLoaderConfig:
        logger.info("Getting data loader config")
        params = self.params.data_loader
        logger.info(f"Data loader config: {params}")
        data_loader_config = DataLoaderConfig(
            train_batch_size=params.train_batch_size,
            valid_batch_size=params.valid_batch_size,
            num_workers=params.num_workers,
            pin_memory=params.pin_memory,
            drop_last=params.drop_last
        )
        logger.info(f"Data loader config created: {data_loader_config}")
        return data_loader_config