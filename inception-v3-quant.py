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
import sys

argv=sys.argv[1:]

device = parse_options(argv)

model_dir = '/inception_v3_quant/'
model_name ='inception_v3_quant.tflite'

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
if device=="llvm":
    desired_layouts = {'qnn.conv2d': ['NCHW', 'default']}
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),relay.transform.ConvertLayout(desired_layouts)])
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

tvm_target = get_tvm_target(device, get_device_type(), get_device_arch(), get_device_attributes())

cpu_target = "llvm"
tvm_targets = tvm.target.Target(tvm_target, host=cpu_target)

cpudevice = tvm.runtime.cpu()

with tvm.transform.PassContext(opt_level=3):
    graph_mod = relay.build(mod, tvm_targets, params=params)

params = graph_mod.get_params()

# Create a runtime executor module
module = graph_executor.GraphModule(graph_mod["default"](cpudevice))

# Feed input data
module.set_input(input_tensor, tvm.nd.array(image_data))

# Feed related params
module.set_input(**params)

ftimer = module.module.time_evaluator("run", cpudevice, number=1, repeat=10)
prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
print("%-20s %-7s %-19s (%s)" % (model_name, device, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))
print(tvm_target)

