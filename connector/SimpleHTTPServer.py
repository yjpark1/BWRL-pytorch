#!/usr/bin/env python

import logging.handlers
from main import GlobalVariable as gvar

logger = logging.getLogger('Main')

# If the predefined port is unavailable, it must be assigned to the other process or application.
# we then should remove the process or application by the following commands.
# netstat -ano | findstr :8888
# taskkill /PID 27932 /F

import tornado.websocket
import time
from tornado import iostream
from tornado.options import options

try:
    from urlparse import urljoin, urldefrag
except ImportError:
    from urllib.parse import urljoin, urldefrag

#define("port", default=8888, help="run on the given port", type=int)
#logging -> {debug, info, warning, error, critical}

# we gonna store clients in dictionary..
clients = dict()

class Connection(object):
    def __init__(self, connection):
        self.stream = iostream.IOStream(connection)

    def _read(self):
        self.write("return values...")
        logger.info("in>>>>>>>>>>>>>>>>>>>>")

    def _eol_callback(self, data):
        logger.info("callback function...")
        self.handle_data(data)

    def connection_ready(sock, fd, events):
        while True:
            connection, address = sock.accept()

class CommunicationHandler(Connection):
    """Put your app logic here"""
    def handle_data(self, data):
        self.stream.write(data)
        self._read()

class MainHandler(tornado.web.RequestHandler):
    instr = None
    outstr = None

    def post(self):
        print(self.request.body)
        gvar.token_deque.append(self.request.body)
        gvar.service_flag = 1
        gvar.flag_restart = self.request.headers.get('isRestarted')
        self.flag_restart = gvar.flag_restart

        while True:
            time.sleep(1e-2)
            if gvar.release_action:
                break

        gvar.release_action = False
        sendAction = self.set_action()
        self.write(sendAction)

    def on_message(self, message):
        # logger.debug("Client %s received a message : %s" % (self.id, message))
        pass

    def on_close(self):
        if self.id in clients:
            del clients[self.id]

    def set_action(self):
        while True:
            if(gvar.action != None):
                self.outstr = gvar.action
                gvar.action = None
                # logger.critical("action exists")
                break;
            else:
                # logger.critical("no action")
                time.sleep(1e-5)

        return self.outstr

app = tornado.web.Application(
    handlers=[
        (r'/', MainHandler),
    ], debug=True
)

if __name__ == "__main__":
    tornado.options.parse_command_line()
    type = gvar.SERVER_TYPE
    if type == "single_proc":
        http_server = tornado.httpserver.HTTPServer(app)
        http_server.listen(gvar.port)
        tornado.ioloop.IOLoop.instance().start()
    elif type == "multi_proc":
        server = tornado.httpserver.HTTPServer(app)
        server.bind(gvar.port)
        server.start(0)  # Forks multiple sub-processes
        tornado.ioLoop.current().start()
    elif type == "advanced_multi_proc":
        sockets = tornado.netutil.bind_sockets(gvar.PORT)
        tornado.process.fork_processes(0)
        server = tornado.httpserver.HTTPServer(app)
        server.add_sockets(sockets)
        tornado.ioLoop.current().start()
    else:
        assert "invalid type"
