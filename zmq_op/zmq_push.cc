//File: zmq_push.cc
//Author: Teyen Liu <teyen.liu@gmail.com>

#include <string>
#include <memory>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include "../includes/zmq.hpp"
#include "../includes/tensor_msg.hpp"

using namespace std;
using namespace tensorflow;


REGISTER_OP("ZMQPush")
    .Attr("T: {float, double, int32}")
    .Attr("address: string")
    .Input("input: T")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful()
    .Doc(R"doc(
Produce list of tensors when reading from ZMQ pipe.

Data serialized by a custom `tensor_msg` within C++ applications can be sent
over a socket directly to a TF graph. Length of list of tensors has to be
known beforehand. This only supports tf.float32, tf.int32, tf.double32.

address: string containing the addess of the socket.
)doc");


class zmq_pipe
{
public:
  zmq_pipe(std::string address) : context(1), socket(context, ZMQ_PUSH){
    socket.connect(address);
  }
  zmq::context_t context;
  zmq::socket_t socket;
};


template <typename T>
class ZMQPush: public OpKernel {
 public:
  explicit ZMQPush(OpKernelConstruction* context) : OpKernel(context) {
    // get pipe name
    string address;
    OP_REQUIRES_OK(context, context->GetAttr("address", &address));
    // setup ZMQ connection
    pipe.reset(new zmq_pipe(address));

  }

  void Compute(OpKernelContext* context) override {

    std::vector<tensor_msg > tensors;

    // get input tensor number
    DCHECK_LE(1, context->num_inputs());

    for (int i = 0; i < context->num_inputs(); i++)
    {
      // get tensor
      const Tensor& my_tensor = context->input(i);
      TensorShape my_shape = my_tensor.shape();
      int dims = my_tensor.dims();

      // this is for the argument of tensor_msg
      std::vector<unsigned int> d_shape;
      for (int d = 0; d < dims; d++)
      {
        d_shape.push_back(my_shape.dim_size(d));
        VLOG(0) << "tensor:" << i << ", dims: " << d << ", elements:" << my_shape.dim_size(d);
      }
      VLOG(0) <<  "type:" << my_tensor.dtype();

      //FIXME: how to detect the data type from input tensor more efficiently
      auto input_data = my_tensor.flat<T>();
      int N = input_data.size();
      T *zmq_data = new T[N];
      for (int n = 0; n < N; n++)
      {
        zmq_data[n] = input_data(n);
      }
      tensors.push_back(tensor_msg(d_shape, zmq_data));
    }
    
    // encode
    msgpack::sbuffer sbuf;
    msgpack::pack(sbuf, tensors);

    zmq::message_t *msg = new zmq::message_t(sbuf.size());
    memcpy(msg->data(), sbuf.data(), sbuf.size());

    // message reader
    pipe->socket.send(*msg);
    delete msg;
    
  }

private:

  std::string address;
  unique_ptr<zmq_pipe> pipe;

};

//#define CPU_KERNEL(type)
REGISTER_KERNEL_BUILDER(Name("ZMQPush").Device(DEVICE_CPU).TypeConstraint<float>("T"), ZMQPush<float>)
REGISTER_KERNEL_BUILDER(Name("ZMQPush").Device(DEVICE_CPU).TypeConstraint<int32>("T"), ZMQPush<int32>)
REGISTER_KERNEL_BUILDER(Name("ZMQPush").Device(DEVICE_CPU).TypeConstraint<double>("T"), ZMQPush<double>)
//CPU_KERNEL(int32)
//CPU_KERNEL(float)
