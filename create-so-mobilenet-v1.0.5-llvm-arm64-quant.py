import os
import pstats
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

def extract(path):
    import tarfile
    if path.endswith("tgz") or path.endswith("gz"):
        dir_path = os.path.dirname(path)
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError('Could not decompress the file: ' + path)

def load_test_image(dtype='float32'):
    image_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
    image_path = download_testdata(image_url, 'cat.png', module='data')
    resized_image = Image.open(image_path).resize((128, 128))

    #image_data = np.asarray(resized_image).astype("float32")
    image_data = np.asarray(resized_image).astype("uint8")

    # Add a dimension to the image so that we have NHWC format layout
    image_data = np.expand_dims(image_data, axis=0)
    
    print('input', image_data.shape)
    return image_data

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

image_data = load_test_image()

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

# Build the module against to x86 CPU
target = "llvm -mattr=+neon"
#target = "arm_cpu -mtriple=armv7a-linux-gnueabihf -mattr=+neon,+vfp4,+thumb2"

cpudevice = tvm.runtime.cpu()
#ctx = tvm.context(str(target), 0)
ctx = tvm.runtime.context("cpu")

with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod, target, params=params)

lib.save("mob-1.0.5-128-arm64-quant.o")
tvm.contrib.cc.create_shared("mob-1.0.5-128-arm64-quant.so", ["mob-1.0.5-128-arm64-quant.o"])

# Create a runtime executor module
module = graph_runtime.create(graph, lib, cpudevice)

with open("./mob_graph.json", "w") as fo:
    fo.write(graph)

with open("./mob_param.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))


print("shared library created\n")
