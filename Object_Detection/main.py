from src.pipeline.stage_05_data_loader import DataLoaderPipeline
from src.pipeline.stage_04_dataset import DatasetPipeline
from src.pipeline.stage_03_data_transformation import DataTransformationPipeline
from src.utils.logging_setup import logger
from src.config.configuration import ConfigurationManager
from src.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.pipeline.stage_02_data_validation import DataValidationPipeline
if __name__ == '__main__':
    try:
        config_manager = ConfigurationManager()
        
        # --- Data Ingestion Stage ---
        STAGE_NAME = "Data Ingestion Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion_pipeline = DataIngestionPipeline(config=config_manager)
        data_ingestion_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Data Validation Stage ---
        STAGE_NAME = "Data Validation Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_validation_pipeline = DataValidationPipeline(config=config_manager)
        data_validation_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Data Transformation Stage ---
        STAGE_NAME = "Data Transformation Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_transformation_pipeline = DataTransformationPipeline(config=config_manager)
        train_transforms, valid_transforms = data_transformation_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")


        STAGE_NAME = "Dataset Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        dataset_pipeline = DatasetPipeline(config=config_manager)
        train_dataset = dataset_pipeline.run_pipeline(subset="train", transforms=train_transforms)
        valid_dataset = dataset_pipeline.run_pipeline(subset="valid", transforms=valid_transforms)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")


        STAGE_NAME = "Data Loader Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_loader_pipeline = DataLoaderPipeline(config=config_manager)
        train_loader = data_loader_pipeline.run_pipeline(train_dataset, shuffle=True)
        valid_loader = data_loader_pipeline.run_pipeline(valid_dataset, shuffle=False)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.error(f"Error occurred during {STAGE_NAME} stage: {e}")
        raise e