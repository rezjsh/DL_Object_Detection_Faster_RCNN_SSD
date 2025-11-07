import os
from src.entity.config_entity import DataValidationConfig
from src.utils.logging_setup import logger

class DataValidation:
    """
    Component to validate the structure and integrity of the ingested dataset.
    """
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        """
        Checks if the required split directories and files exist in the dataset folder.
        Returns True if all required files/directories exist, else False.
        """
        
        try:
            validation_status = True
            
            # 1. Check for required splits/files
            logger.info("Checking for required files and directories...")
            for file_or_dir in self.config.required_files:
                path_to_check = os.path.join(self.config.data_dir, file_or_dir)
                if not os.path.exists(path_to_check):
                    logger.error(f"Validation FAILED: Required file/directory missing: {path_to_check}")
                    validation_status = False


            # 2. Save validation status
            logger.info("Saving validation status...")
            with open(self.config.status_file, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            
            if validation_status:
                logger.info("Validation PASSED: All required files/directories are present.")
                
            return validation_status
            
        except Exception as e:
            logger.error(f"Error during data validation: {e}", exc_info=True)
            raise e