# -*- coding: utf-8 -*-

import logging

class cLogger():
    global LOGGER

    @staticmethod
    def getLogger(loggerName='not_init'):
        LOGGER = logging.getLogger('Main')
        if loggerName is 'init':
            fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
            fileHandler = logging.FileHandler('./myLoggerTest.log', 'w', 'utf-8')
            streamHandler = logging.StreamHandler()
            fileHandler.setFormatter(fomatter)
            streamHandler.setFormatter(fomatter)
            LOGGER.addHandler(fileHandler)
            # LOGGER.addHandler(streamHandler)
            LOGGER.setLevel(logging.DEBUG)
            logging.basicConfig(filename='./test.log',level=logging.DEBUG)
        else:
            pass
        return LOGGER
