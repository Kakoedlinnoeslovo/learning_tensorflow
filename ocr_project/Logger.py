import logging
from Config import TEMP_DIR


class Logger:
    def __init__(self, filename = TEMP_DIR + "log.txt"):
        format = "%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s"
        logging.basicConfig(level = logging.DEBUG,
                            format = format,
                            filename = filename)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(format)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)



if __name__ == '__main__':
    Logger()
    logging.info("loading data")



