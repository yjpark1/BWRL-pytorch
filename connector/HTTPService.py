# -*- coding: utf-8 -*-
import tornado
import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.websocket

from connector.SimpleHTTPServer import MainHandler
from util.CustomLog import cLogger

from main import GlobalVariable as gvar

logger = cLogger.getLogger()

def __init__(self):
    pass

def connect(a=None):
    app = tornado.web.Application(
        handlers=[
            (r'/', MainHandler),
        ], debug=True)

    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(gvar.PORT)

    tornado.ioloop.IOLoop.instance().start()

    #logger.debug("===============================================")
    #logger.debug("\t>>> 2. Initialize the http server....")
    #logger.debug("\tType: Synchronous and blocking I/O")
    #logger.debug("=============================================\n")