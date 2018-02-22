mkdir -p build
# build simple writer
echo "build simple writer"
CFLAGS=`pkg-config --cflags libzmq msgpack` 
LFLAGS=`pkg-config --libs libzmq msgpack`

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

g++ write.cpp $CFLAGS $LFLAGS -o build/write -std=c++11
g++ read.cpp $CFLAGS $LFLAGS -o build/read -std=c++11
g++ dump.cpp $CFLAGS $LFLAGS -o build/dump -std=c++11

# build zmq_op for tensorflow
echo "build zmq_pull_op for tensorflow"
g++ -std=c++11 -shared zmq_op/zmq_pull.cc -o zmq_op/zmq_pull.so -fPIC -I $CFLAGS -L $LFLAGS ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2

echo "build zmq_push_op for tensorflow"
#echo "g++ -std=c++11 -shared zmq_op/zmq_push.cc -o zmq_op/zmq_push.so -fPIC -I $CFLAGS -L $LFLAGS ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2"
g++ -std=c++11 -shared zmq_op/zmq_push.cc -o zmq_op/zmq_push.so -fPIC -I $CFLAGS -L $LFLAGS ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
