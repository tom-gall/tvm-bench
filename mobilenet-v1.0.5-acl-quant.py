import os
import numpy as np
import tvm
from PIL import Image
from tvm import te
from tvm.contrib import graph_runtime
from tvm import relay
from tvm.runtime import container
from tvm.runtime import vm as vm_rt
from tvm.relay import testing
from tvm.relay import vm
from tvm.relay.op.contrib import arm_compute_lib
from tvm.contrib.download import download_testdata
from util import load_test_image,build_module, update_lib, get_cpu_op_count, download_model_zoo,parse_options
import sys

argv=sys.argv[1:]

device = parse_options(argv)

model_dir = '/mobilenet-v1.0.5-128quant/'
model_name ='mobilenet_v1_0.5_128_quant.tflite'

model_dir = download_model_zoo(model_dir, model_name)

tflite_model_file = os.path.join(model_dir, model_name)
tflite_model_buf = open(tflite_model_file, "rb").read()

# Get TFLite model from buffer
try:
    import tflite
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

dtype="uint8"
width=128
height=128
image_data = load_test_image(dtype, width, height)

input_tensor = "input"
input_shape = (1, 128, 128, 3)
#input_shape = (1, 3, 224, 224)
#input_dtype = "float32"
input_dtype = "uint8"

# Parse TFLite model and convert it to a Relay module
mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict={input_tensor: input_shape},
                                         dtype_dict={input_tensor: input_dtype})

# Build the module for a Arm CPU
if device in ("llvm"):
    target = "llvm -mcpu=thunderxt88 -mtriple=aarch64-unknown-linux-gnu -mattr=+neon,+crc,+lse"
else:
    target = "llvm -device=arm_cpu -mcpu=thunderxt88 -mtriple=aarch64-unknown-linux-gnu -mattr=+neon,+crc,+lse"

tvm_targets = tvm.target.Target(target)
cpu_target = "llvm"
target_host=cpu_target

cpudevice = tvm.runtime.cpu()
ctx = tvm.runtime.context("cpu")

enable_acl=True
tvm_ops=42
acl_partitions=17
atol=0.002
rtol=0.01

try:
    lib = build_module(mod, target, params, enable_acl, tvm_ops, acl_partitions)
except Exception as e:
    err_msg = "The module could not be built.\n"
    raise Exception(err_msg)

gen_module = graph_runtime.GraphModule(lib["default"](cpudevice))
gen_module.set_input(input_tensor, tvm.nd.array(image_data))
gen_module.run()


ftimer = gen_module.module.time_evaluator("run", ctx, number=1, repeat=10)
prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
print("acl %-20s %-7s %-19s (%s)" % (model_name, device, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))



#with tvm.transform.PassContext(opt_level=3):
#    graph_mod = relay.build(mod, tvm_targets, params=params,target_host=target_host)

#lib = graph_mod.get_lib()
#params = graph_mod.get_params()
#graph = graph_mod.get_json()

# Create a runtime executor module
#module = graph_runtime.create(graph, lib, cpudevice)

# Feed input data
#module.set_input(input_tensor, tvm.nd.array(image_data))

# Feed related params
#module.set_input(**params)

#ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
#prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
#print("%-20s %-19s (%s)" % (model_name, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))

# Run
#module.set_input(input_tensor, tvm.nd.array(image_data))
#module.set_input(**params)
#module.run()

# Get output
#tvm_output = module.get_output(0).asnumpy()

# Load label file
#label_file_url = ''.join(['https://raw.githubusercontent.com/',
#                          'tensorflow/tensorflow/master/tensorflow/lite/java/demo/',
#                          'app/src/main/assets/',
#                          'labels_mobilenet_quant_v1_224.txt'])
#label_file = "labels_mobilenet_quant_v1_224.txt"
#label_path = download_testdata(label_file_url, label_file, module='data')
# List of 1001 classes
#with open(label_path) as f:
#    labels = f.readlines()

# Convert result to 1D data
#predictions = np.squeeze(tvm_output)

# Get top 1 prediction
#prediction = np.argmax(predictions)

# Convert id to class name and show the result
#print("The image prediction result is: id " + str(prediction) + " name: " + labels[prediction])
