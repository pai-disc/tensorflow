/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/tsl/platform/cuda_libdevice_path.h"

#include <stdlib.h>

#include <string>
#include <vector>

#include "tensorflow/tsl/platform/platform.h"

#if !defined(PLATFORM_GOOGLE)
#include "third_party/gpus/cuda/cuda_config.h"
#endif
#include "tensorflow/tsl/platform/logging.h"

namespace tsl {

std::vector<std::string> CandidateCudaRoots() {
  std::vector<string> candidates = {
    TF_CUDA_TOOLKIT_PATH, string("/usr/local/cuda")
  };
  if (char* pstr = getenv("TAO_CUDA_PATH")) {
    candidates.push_back(pstr);
  }
  return candidates;
}

bool PreferPtxasFromPath() { return true; }

}  // namespace tsl
