// Author: Patrick Wieschollek <mail@patwie.com>
#include "includes/zmq.hpp"
#include "includes/tensor_msg.hpp"
#include <msgpack.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <thread>


int main () {

    std::vector<tensor_msg > tensors;

    // create a tensor of 4x4 floats
    float *image_data = new float[4*4];
    for (int i = 0; i < 4*4; ++i)
      image_data[i] = (float) i;

    int *label_data = new int[1];
    label_data[0] = 42;
    
    tensors.push_back(tensor_msg({4, 4}, image_data));
    tensors.push_back(tensor_msg({1}, label_data));

    // encode
    msgpack::sbuffer sbuf;
    msgpack::pack(sbuf, tensors);

    //  Prepare our context and socket
    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_PUSH);
    socket.connect ("ipc:///tmp/ipc-socket-0");

    while (true) {
        zmq::message_t *msg = new zmq::message_t(sbuf.size());
        memcpy (msg->data (), sbuf.data(), sbuf.size());

        // message reader
        std::cout << "Prepare to send message ..." << std::endl;
        socket.send(*msg);
        std::cout << "Send number of tensors: "<< msg->size() << " ..." << std::endl;

        delete msg;

        //  wait
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    return 0;
}