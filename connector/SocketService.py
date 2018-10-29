import socket
import sys
import msgpack
import time
from main import GlobalVariable as gvar

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = ('localhost', 5555)
print(sys.stderr, 'starting up on %s port %s'.format(server_address))
sock.bind(server_address)

# Listen for incoming connections
sock.listen(5)

BUFF_SIZE = 4096  # 4 KiB

outstr = None

def recvall(sock):

    data = b''

    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        print('length : ', len(part))
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
            break

    return data
def __init__(self):
    #logger.debug("test>>>>>>>>>>>>>>>>>>>>>>>>")
    pass

def connect(a=None):
    while True:
        # Wait for a connection
        print(sys.stderr, 'waiting for a connection')
        connection, client_address = sock.accept()

        try:
            print(sys.stderr, 'connection from', client_address)

            # Receive the data in small chunks and retransmit it
            while True:
                header = connection.recv(5)
                d_header = msgpack.unpackb(header)

                is_action_needed = d_header[0];
                size_of_data = d_header[1]
                print('action needed : {}'.format(is_action_needed))
                print('data size : {}'.format(size_of_data))

                #data = connection.recv(4096)
                data = recvall(connection)

                #print(sys.stderr, 'received "%s"' % data)
                if data:
                    # print('trying to unpack msgpack msg received from client..')
                    print('-------------------------msg----------------------------')
                    msgpack_data = msgpack.unpackb(data)
                    print('type(msgpack_data) : {}'.format(type(msgpack_data)))
                    gvar.token_deque.append(msgpack_data)
                    print('1')


                # TODO : action 분기처리
                if is_action_needed == 1:

                    print('2')
                    gvar.service_flag = 1
                    gvar.action = set_action()

                    print('len(gvar.action) :' , len(gvar.action))

                    # TODO: action을 client에 전송!
                    connection.send(gvar.action)
                else:
                    print('3')
                    gvar.service_flag = 0
                    pass

        finally:
            # Clean up the connection
            connection.close()

def set_action():
    outstr = None
    while True:
        if (gvar.action != None):
            outstr = gvar.action
            # logger.critical("action exists")
            break;
        else:
            # logger.critical("no action")
            time.sleep(0.)

    return outstr