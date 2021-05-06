import os
import numpy as np
import tvm
from PIL import Image
from tvm import te, autotvm
from tvm.contrib import graph_executor
from tvm import relay
from tvm.runtime import container
from tvm.runtime import vm as vm_rt
from tvm.relay import testing
from tvm.relay import vm
from tvm.contrib.download import download_testdata
from util import load_test_image, download_model_zoo,parse_options, get_device_arch, get_device_attributes, get_device_type, get_tvm_target, parse_cmd_options
import sys

argv=sys.argv[1:]

device, logfile = parse_cmd_options(argv)

model_dir ="mobilebert_v1/"
model_name = "mobilebert_1_default_1.tflite"

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

input_dtype="float32"
batch = 1
seq_length = 384

#Random inputs for BERT network. Previous example had segment_ids (valid_length) as just (batch,) which was causing errors. 
inputs = np.random.randint(0, 2000, size=(batch, seq_length)).astype(input_dtype)
token_types = np.random.uniform(size=(batch, seq_length)).astype(input_dtype)
valid_length = np.random.uniform(size=(batch, seq_length)).astype(input_dtype)

input_tensors = ['input_ids', 'input_mask', 'segment_ids']
input_shape = (batch, seq_length)

#Forming shape_dict and dtype_dict according to tflite model
shape_dict = {input_tensors[0]: input_shape, input_tensors[1]: input_shape, input_tensors[2]: input_shape}
dtype_dict = {input_tensors[0]: input_dtype, input_tensors[1]: input_dtype, input_tensors[2]: input_dtype}

mod, params = relay.frontend.from_tflite(tflite_model, shape_dict)
tvm_target = get_tvm_target(device, get_device_type(), get_device_arch(), get_device_attributes())

tvm_targets = tvm.target.Target(tvm_target)
cpu_target = "llvm"
target_host = cpu_target

cpudevice = tvm.runtime.cpu()

if logfile is not None:
    with autotvm.apply_history_best(logfile):
        with tvm.transform.PassContext(opt_level=3, required_pass=["FastMath"]):
            graph_mod = relay.build(mod, tvm_targets, params=params,target_host=target_host)
else:
    with tvm.transform.PassContext(opt_level=3, required_pass=["FastMath"]):
        graph_mod = relay.build(mod, tvm_targets, params=params,target_host=target_host)

lib = graph_mod.get_lib()
params = graph_mod.get_params()
graph = graph_mod.get_json()

# Create the executor and set the parameters and inputs
module = graph_executor.create(graph, lib, tvm.cpu())
# Feeding input data
module.set_input(input_ids=inputs, input_mask=token_types, segment_ids=valid_length)
module.set_input(**params)

ftimer = module.module.time_evaluator("run", cpudevice, number=1, repeat=10)
prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
print("%-20s %-7s %-19s (%s)" % (model_name, device, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))
print(tvm_target)
