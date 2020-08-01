import os
import numpy as np
import tvm
from PIL import Image
from tvm import te
# from tvm.contrib import graph_runtime
from tvm import relay
from tvm.runtime import container
from tvm.runtime import vm as vm_rt
from tvm.relay import testing
from tvm.relay import vm
from tvm.contrib.download import download_testdata
from tvm import autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_runtime as runtime
import logging

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
    
    #image_data = np.transpose(image_data, (0, 3, 1, 2))

    print('input', image_data.shape)
    return image_data

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size)
    elif name == 'squeezenet_v1.1':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape

def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    #if os.path.exists(tmp_log_file):
    #    os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)
                       ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    # os.remove(tmp_log_file)

def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
    target_op = [relay.op.get("nn.conv2d"),]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)


logging.getLogger('autotvm').setLevel(logging.DEBUG)
target = tvm.target.create("llvm -model=cortex-a7 -mtriple=arm-linux-gnueabihf -mattr=+neon,+vfp4,+thumb2 -mcpu=cortex-a7")
#target = tvm.target.create('llvm -device=arm_cpu -mtriple=aarch64-linux-gnu')

# Also replace this with the device key in your tracker
device_key = 'stm32mp1'

# Set this to True if you use android phone
use_android = False

#### TUNING OPTION ####
network = 'MobileNetv1.0.5'
log_file = "%s.%s.log" % (device_key, network)
dtype = 'uint8'
graph_opt_sch_file = "%s_graph_opt.log" % network

tuning_option = {
    'log_filename': log_file,

    #'tuner': 'random',
    'tuner': 'xgb',
    'n_trial': 1600,
    'early_stopping': None,
    #'try_spatial_pack_depthwise':True,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(
            build_func='ndk' if use_android else 'default'),
        runner=autotvm.RPCRunner(
            device_key, host='0.0.0.0', port=9190,
            number=250,
            timeout=1000,
        ),
    ),
}
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
input_shape = (30, 128, 128, 3)
#input_shape = (1, 3, 224, 224)
#input_dtype = "float32"
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
#target = "llvm -model=cotex-a7 -mattr=+neon,+vfp4,+thumb2 -mcpu=cortex-a7"
#target = "arm_cpu -mtriple=armv7a-linux-gnueabihf -mattr=+neon,+vfp4,+thumb2"

#t = tvm.target.arm_cpu(options="-device=arm_cpu -mtriple=armv7a-linux-gnueabihf -mattr=+neon,+vfp4,+thumb2")

cpudevice = tvm.runtime.cpu()
#ctx = tvm.context(str(target), 0)
ctx = tvm.runtime.context("cpu")

#with relay.build_config(opt_level=3):
#    graph, lib, params = relay.build(mod, target, params=params)


# Create a runtime executor module
#module = graph_runtime.create(graph, lib, cpudevice)

network ="mobilenet"

#mod, params, input_shape, _ = get_network(network, batch_size=1)
tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                          params=params,
                                          ops=(relay.op.get("nn.conv2d"),))

# run tuning tasks
print("Tuning...")
tune_tasks(tasks, **tuning_option)
# tune_graph(mod["main"], input_shape, log_file, graph_opt_sch_file)

# compile kernels with history best records
with autotvm.apply_history_best(log_file):
    print("Compile...")
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build_module.build(
            mod, target=target, params=params)

    # export library
    tmp = tempdir()
    filename = "net.tar"
    lib.export_library(tmp.relpath(filename))

    # upload module to device
    print("Upload...")
    remote = autotvm.measure.request_remote(device_key, '0.0.0.0', 9190,
                                                timeout=10000)
    remote.upload(tmp.relpath(filename))
    rlib = remote.load_module(filename)

    # upload parameters to device
    ctx = remote.context(str(target), 0)
    module = runtime.create(graph, rlib, ctx)
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    #module.set_input(input_tensor, tvm.nd.array(image_data))
    module.set_input(input_tensor, data_tvm)
    module.set_input(**params)

    # evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=10, repeat=3)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
          (np.mean(prof_res), np.std(prof_res)))





# Feed input data
#module.set_input(input_tensor, tvm.nd.array(image_data))

# Feed related params
#module.set_input(**params)

#ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
#prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
#print("%-20s %-19s (%s)" % (model_name, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))

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
