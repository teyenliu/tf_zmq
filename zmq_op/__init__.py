import tensorflow as tf
import os

__all__ = ['pull', 'push']

path = os.path.join(os.path.dirname(__file__), 'zmq_pull.so')
_zmq_pull_module = tf.load_op_library(path)
pull = _zmq_pull_module.zmq_pull

path = os.path.join(os.path.dirname(__file__), 'zmq_push.so')
_zmq_push_module = tf.load_op_library(path)
push = _zmq_push_module.zmq_push

