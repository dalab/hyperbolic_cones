import logging
import os
import numpy as np

### Uncomment this line if you want to see the output at stdout as well.
# logging.basicConfig(level=logging.INFO)

def setup_logger(name_logfile, also_stdout=False):
    name_logfile = name_logfile.replace(';', '#')
    name_logfile = name_logfile.replace(':', '_')
    current_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    logger = logging.getLogger(name_logfile)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    fileHandler = logging.FileHandler(os.path.join(current_directory, 'logs/', name_logfile), mode='w')
    fileHandler.setFormatter(formatter)
    if also_stdout:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)
    if also_stdout:
        logger.addHandler(streamHandler)
    return logger










