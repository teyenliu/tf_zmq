import tensorflow as tf
import zmq_op
import signal
import sys
def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


output = zmq_op.pull('ipc:///tmp/ipc-socket-0', [tf.float32])
with tf.Session() as sess:
    while(1):
        print(sess.run([output]))
        signal.pause()
