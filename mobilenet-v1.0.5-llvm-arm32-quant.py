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

model_url = "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz"

# Download model tar file and extract it to get mobilenet_v1_1.0_224.tflite
#model_path = download_testdata(model_url, "mobilenet_v1_1.0_224.tgz", module=['tf', 'official'])
#model_dir = os.path.dirname(model_path)
model_dir = './mobilenet-v1.0.5-128quant/'
#extract(model_path)
model_name ='mobilenet_v1_0.5_128_quant.tflite'
# Now we can open mobilenet_v1_1.0_224.tflite
#tflite_model_file = os.path.join(model_dir, "mobilenet_v1_1.0_224.tflite")
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
input_dtype = "uint8"

# Parse TFLite model and convert it to a Relay module
mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict={input_tensor: input_shape},
                                         dtype_dict={input_tensor: input_dtype})

#desired_layouts = {'nn.conv2d': ['NCHW', 'default']}
#seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
#                                relay.transform.ConvertLayout(desired_layouts)])
#with tvm.transform.PassContext(opt_level=3):
#    mod = seq(mod)

# Build the module for ARM CPU
target = "llvm -mtriple=armv7a-linux-gnueabihf -mattr=+neon-vfp4,+thumb2"
tvm_targets = tvm.target.create(target)
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
module = graph_runtime.create(graph, lib, cpudevice)

# Feed input data
module.set_input(input_tensor, tvm.nd.array(image_data))

# Feed related params
module.set_input(**params)

ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
print("%-20s %-19s (%s)" % (model_name, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))

# Run
#module.run()

# Get output
#tvm_output = module.get_output(0).asnumpy()

# Load label file
#label_file_url = ''.join(['https://raw.githubusercontent.com/',
#                          'tensorflow/tensorflow/master/tensorflow/lite/java/demo/',
#                          'app/src/main/assets/',
#                          'labels_mobilenet_quant_v1_224.txt'])
#label_file = "labels_mobilenet_quant_v1_224.txt"
#label_file = "labels.txt"
#label_path = download_testdata(label_file_url, label_file, module='data')
#label_path = "./labels.txt"

# List of 1001 classes
#with open(label_path) as f:
#    labels = f.readlines()

# Convert result to 1D data
#predictions = np.squeeze(tvm_output)

#top_k = predictions.argsort()[-5:][::-1]
#for node_id in top_k:
    #human_string = lsnode_lookup.id_to_string(node_id)
    #print(labels[node_id])

# Get top 1 prediction
#prediction = np.argmax(predictions)

# Convert id to class name and show the result
#print("The image prediction result is: id " + str(prediction) + " name: " + labels[prediction])
