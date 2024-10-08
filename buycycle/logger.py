import logging
from pythonjsonlogger import jsonlogger


class AppJsonFormatter(jsonlogger.JsonFormatter):
    def __init__(self, environment, ab, app_name, app_version, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.environment = environment
        self.ab = ab
        self.app_name = app_name
        self.app_version = app_version
    def process_log_record(self, log_record):
        log_record['environment'] = self.environment
        log_record['ab'] = self.ab
        log_record['app_name'] = self.app_name
        log_record['app_version'] = self.app_version
        return super().process_log_record(log_record)
class Logger:
    @staticmethod
    def configure_logger(environment, ab, app_name, app_version, log_level=logging.INFO):
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        logger.propagate = False  # Prevent log messages from being duplicated
        # Check if handlers are already configured (to avoid adding multiple handlers)
        if not logger.handlers:
            log_handler = logging.StreamHandler()
            formatter = AppJsonFormatter(environment, ab, app_name, app_version)
            log_handler.setFormatter(formatter)
            logger.addHandler(log_handler)
        return logger


