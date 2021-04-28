import os
import numpy as np
import tvm
from PIL import Image
from tvm import te
from tvm.contrib import graph_executor
from tvm import relay
from tvm.runtime import container
from tvm.runtime import vm as vm_rt
from tvm.relay import testing
from tvm.relay import vm
from tvm.contrib.download import download_testdata
from util import load_test_image, download_model_zoo,parse_options, get_device_arch, get_device_attributes, get_device_type, get_tvm_target
from util import build_module, update_lib, get_cpu_op_count
import sys

argv=sys.argv[1:]

device = parse_options(argv)

model_dir = '/inception_v3_2018_04_27/'
model_name ='inception_v3.tflite'

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

dtype="float32"
width=299
height=299
image_data = load_test_image(dtype, width, height)

input_tensor = "input"
input_shape = (1, 299, 299, 3)
input_dtype = dtype

# Parse TFLite model and convert it to a Relay module
mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict={input_tensor: input_shape},
                                         dtype_dict={input_tensor: input_dtype})
desired_layouts = {'nn.conv2d': ['NCHW', 'default']}
seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),relay.transform.ConvertLayout(desired_layouts)])
with tvm.transform.PassContext(opt_level=3):
    mod = seq(mod)

# Build the module for arm CPU
tvm_target = get_tvm_target(device, get_device_type(), get_device_arch(), get_device_attributes())

tvm_targets = tvm.target.Target(tvm_target)
cpu_target = "llvm"
target_host=cpu_target

cpudevice = tvm.runtime.cpu()

enable_acl=True
tvm_ops=506
acl_partitions=96
atol=0.002
rtol=0.01

try:
    lib = build_module(mod, tvm_target, params, enable_acl, tvm_ops, acl_partitions)
except Exception as e:
    err_msg = "The module could not be built.\n"
    #if config:
    #    err_msg += f"The test failed with the following parameters: {config}\n"
    err_msg += str(e)
    raise Exception(err_msg)


gen_module = graph_executor.GraphModule(lib["default"](cpudevice))
gen_module.set_input(input_tensor, tvm.nd.array(image_data))


ftimer = gen_module.module.time_evaluator("run", cpudevice, number=1, repeat=10)
prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
print("acl %-20s %-7s %-19s (%s)" % (model_name,device, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))
print(tvm_target)
