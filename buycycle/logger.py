import logging
from pythonjsonlogger import jsonlogger
from kafka.producer import KafkaProducer


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


class KafkaLogger(Logger):
    def __init__(self, environment: str, ab: str, app_name: str, app_version: str, topic: str, bootstrap_servers: str, log_level=logging.INFO):
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
        self.topic = topic
        self.logger = self.configure_logger(environment, ab, app_name, app_version, log_level)

    def send_log(self, level, message, extra):
        try:
            log_record = self.logger.makeRecord(self.logger.name, level, None, None, message, None, None, extra={'extra': extra})
            log_json = self.logger.handlers[0].formatter.format(log_record)
            self.producer.send(self.topic, log_json.encode())
        except Exception as e:
            # Handle the exception (e.g., print to stderr, fallback to another logger, etc.)
            print(f"Failed to send log to Kafka: {e}")
    def debug(self, message, extra=None):
        self.send_log(logging.DEBUG, message, extra)

    def info(self, message, extra=None):
        self.send_log(logging.INFO, message, extra)

    def warning(self, message, extra=None):
        self.send_log(logging.WARNING, message, extra)

    def error(self, message, extra=None):
        self.send_log(logging.ERROR, message, extra)
