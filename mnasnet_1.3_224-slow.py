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
from tvm.contrib.download import download_testdata
from util import load_test_image

model_url = "http://download.tensorflow.org/models/mnasnet_1.3_224/mnasnet_1.3_224_1_default_1.tflite"

# Download model tar file and extract it to get mnasnet_1.3_224.tflite
#model_path = download_testdata(model_url, "mnasnet_1.3_224.tgz", module=['tf', 'official'])
#model_dir = os.path.dirname(model_path)
#extract(model_path)
model_dir ="./mnasnet_1.3_224/"
# Now we can open mobilenet_v1_1.0_224.tflite
tflite_model_file = os.path.join(model_dir, "mnasnet_1.3_224.tflite")
tflite_model_buf = open(tflite_model_file, "rb").read()

# Get TFLite model from buffer
try:
    import tflite
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

dtype="float32"
width=224
height=224
image_data = load_test_image(dtype, width, height)

input_tensor = "input"
input_shape = (1, 224, 224, 3)
input_dtype = "float32"

# Parse TFLite model and convert it to a Relay module
mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict={input_tensor: input_shape},
                                         dtype_dict={input_tensor: input_dtype})
# Build the module for arm LLVM
target = "llvm -mtriple=aarch64-unknown-linux-gnu -mattr=+neon"
tvm_targets = tvm.target.Target(target)
cpu_target = "llvm"
target_host=cpu_target

cpudevice = tvm.runtime.cpu()
ctx = tvm.runtime.context("cpu")

with tvm.transform.PassContext(opt_level=3):
    graph_mod = relay.build(mod, tvm_targets, params=params,target_host=target_host)

lib = graph_mod.get_lib()
params = graph_mod.get_params()
graph = graph_mod.get_json()

# Create a runtime executor module
module = graph_runtime.create(graph, lib, tvm.cpu())

# Feed input data
module.set_input(input_tensor, tvm.nd.array(image_data))

# Feed related params
module.set_input(**params)

ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
print("slow %-20s %-19s (%s)" % ("mnasnet_1.3_224.tflite", "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))

