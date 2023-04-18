// Main executable to generate op fuzzers

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
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/cc/framework/cc_op_gen_util.h"
#include "tensorflow/cc/framework/fuzzing/cc_op_fuzz_gen.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
namespace cc_op {
namespace {

void WriteAllFuzzers(const std::string& file_name, bool include_internal,
                     const std::vector<string>& api_def_dirs) {
  OpList ops;
  StatusOr<ApiDefMap> api_def_map =
      LoadOpsAndApiDefs(ops, include_internal, api_def_dirs);

  TF_CHECK_OK(api_def_map.status());
  WriteFuzzers(ops, api_def_map.value());

  Env* env = Env::Default();
  std::unique_ptr<WritableFile> fuzz_file = nullptr;
  auto status = env->NewWritableFile(file_name, &fuzz_file);
  status.Update(fuzz_file->Append(WriteFuzzers(ops, api_def_map.value())));
  status.Update(fuzz_file->Close());
  TF_CHECK_OK(status);
}

}  // namespace
}  // namespace cc_op
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc != 4) {
    for (int i = 1; i < argc; ++i) {
      fprintf(stderr, "Arg %d = %s\n", i, argv[i]);
    }
    fprintf(stderr,
            "Usage: %s out include_internal "
            "api_def_dirs1,api_def_dir2 ...\n"
            "  include_internal: 1 means include internal ops\n",
            argv[0]);
    exit(1);
  }

  bool include_internal = tensorflow::StringPiece("1") == argv[2];
  std::vector<tensorflow::string> api_def_dirs = tensorflow::str_util::Split(
      argv[3], ",", tensorflow::str_util::SkipEmpty());
  tensorflow::cc_op::WriteAllFuzzers(argv[1], include_internal, api_def_dirs);
  return 0;
}
