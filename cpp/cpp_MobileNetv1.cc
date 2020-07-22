/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief Example code on load and run TVM module.s
 * \file cpp_deploy.cc
 */
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <bits/stdc++.h>
#include <chrono>

void Verify(tvm::runtime::Module mod, std::string fname) {
  // Get the function from the module.
  tvm::runtime::PackedFunc f = mod.GetFunction(fname);
  CHECK(f != nullptr);
  // Allocate the DLPack data structures.
  //
  // Note that we use TVM runtime API to allocate the DLTensor in this example.
  // TVM accept DLPack compatible DLTensors, so function can be invoked
  // as long as we pass correct pointer to DLTensor array.
  //
  // For more information please refer to dlpack.
  // One thing to notice is that DLPack contains alignment requirement for
  // the data pointer and TVM takes advantage of that.
  // If you plan to use your customized data container, please
  // make sure the DLTensor you pass in meet the alignment requirement.
  //
  DLTensor* x;
  DLTensor* y;
  int ndim = 1;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int64_t shape[1] = {10};
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);
  for (int i = 0; i < shape[0]; ++i) {
    static_cast<float*>(x->data)[i] = i;
  }
  // Invoke the function
  // PackedFunc is a function that can be invoked via positional argument.
  // The signature of the function is specified in tvm.build
  f(x, y);
  // Print out the output
  for (int i = 0; i < shape[0]; ++i) {
    CHECK_EQ(static_cast<float*>(y->data)[i], i + 1.0f);
  }
  LOG(INFO) << "Finish verification...";
}

int main(void) {
  void * handle=NULL;
  DLTensor* input;

  // Normally we can directly
  tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile("/home/debian/git/tvm-bench/mob-1.0.5-128-arm32-quant.so");
  LOG(INFO) << "dynamic loading from mob-1.0.5-128-arm64-quant.so";

  // load image and or populate data structure

  /* 
  # Feed input data

module.set_input(input_tensor, tvm.nd.array(image_data))

# Feed related params
module.set_input(**params)

# Run
#module.run()

# Get output
#tvm_output = module.get_output(0).asnumpy()
  */

  std::ifstream json_in("/home/debian/git/tvm-bench/mob_graph.json");
  std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
  json_in.close();

  std::ifstream params_in("/home/debian/git/tvm-bench/mob_param.params", std::ios::binary);
  std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
  params_in.close();
    
  int device_type = kDLCPU;
  int device_id = 0;
  // get global function module for graph runtime
  const tvm::runtime::PackedFunc *runtime_create = tvm::runtime::Registry::Get("tvm.graph_runtime.create");
  const tvm::runtime::PackedFunc f = *runtime_create;
  tvm::runtime::Module amod = f(json_data, mod_dylib, device_type, device_id);
  handle = new tvm::runtime::Module(amod);


  TVMByteArray params_arr;
  params_arr.data = params_data.c_str();
  params_arr.size = params_data.length();
  const tvm::runtime::PackedFunc *load_params(tvm::runtime::Registry::Get("load_params"));
  const tvm::runtime::PackedFunc g = amod.GetFunction("load_params");
  CHECK(g != nullptr);
  g(params_arr);

  constexpr int dtype_code = kDLFloat;
  constexpr int dtype_bits = 8;
  constexpr int dtype_lanes = 1;
  constexpr int in_ndim = 4;
  const int64_t in_shape[in_ndim] = {1, 128, 128, 3};
  TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);//
  //TVMArrayCopyFromBytes(input,tensor.data,112*3*112*4);
  tvm::runtime::PackedFunc set_input = amod->GetFunction("set_input");
  set_input("input", input);
  tvm::runtime::PackedFunc run = amod->GetFunction("run");

  auto start = std::chrono::high_resolution_clock::now();

  run();
  auto end = std::chrono::high_resolution_clock::now();

  double time_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  time_elapsed *= 1e-9;

  LOG(INFO) << "it ran  mob-1.0.5-128-arm32-quant.so";
  LOG(INFO) << "time was " << time_elapsed << std::setprecision(9);
  //tvm::runtime::Module.run()
  //Verify(mod_dylib, "addone");
  // For libraries that are directly packed as system lib and linked together with the app
  // We can directly use GetSystemLib to get the system wide library.
  //LOG(INFO) << "Verify load function from system lib";
  //tvm::runtime::Module mod_syslib = (*tvm::runtime::Registry::Get("runtime.SystemLib"))();
  //Verify(mod_syslib, "addonesys");
  return 0;
}
