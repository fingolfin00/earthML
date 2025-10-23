import colorlog, logging

class Logger:
    def __init__(self, log_filename, log_level="info"):
        self.log_level = getattr(logging, log_level.upper())
        self.log_filename = log_filename
        self.log_format_string = ('[%(asctime)s %(levelname)s] %(message)s')
        self.log_format_string_color = (
                                        # '%(log_color)s%(asctime)s [%(levelname)s: %(name)s] %(message)s'
                                        # '%(log_color)s[%(levelname)s]%(reset)s %(name)s: %(message)s'
                                        '%(log_color)s[%(levelname)s]%(reset)s %(message)s'
                                        # '%(funcName)s:%(lineno)d - %(message)s'
        )
        self.logger = self.setup_logger(logging.getLogger(__name__))

    def setup_logger(self, logger: logging.Logger):
        logger.setLevel(self.log_level)
        logger.handlers = []  # Clear existing handlers
        formatter_color = colorlog.ColoredFormatter(
            self.log_format_string_color,
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            },
        )
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(formatter_color)
        logger.addHandler(console_handler)
        formatter = logging.Formatter(self.log_format_string)
        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger
