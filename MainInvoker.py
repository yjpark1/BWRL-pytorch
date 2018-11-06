# -*- coding: utf-8 -*-

import threading
from util.CustomLog import cLogger
from connector.HTTPService import connect
from main import GlobalVariable as gvar
from main.starcraft_modelRL_TD3 import rl_learn

logger = cLogger.getLogger(loggerName='init')

def run():

    threading.Thread(name='RL training', target=rl_learn).start()
    logger.info(">>> RL env starts ... (1/2)")
    threading.Thread(name='Http Server', target=connect).start()
    logger.info(">>> HTTP service starts ... (2/2)")
    logger.info(">>> RL Server is now ready to accept connections on port " + str(gvar.PORT))

if __name__ == '__main__':
    run()