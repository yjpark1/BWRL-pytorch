# -*- coding: utf-8 -*-
######################################################
#
# Configurations for RL env
#
######################################################
import collections

##### Constant
# TODO
cuda_device = "0"

##### Variable

######################################################
#
# Configurations for HTTP Server
#
######################################################

##### Constant
SERVER_TYPE = "single_proc"
PORT = 11111


##### Variable
#token = "" # message received from client
token_deque = collections.deque(maxlen=200)
service_flag = 0 # deprecated
action = None  # message returned by RL env
flag_restart = 0
release_action = True

