// Copyright 2022 The TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_JITRT_CUSTOM_CALLS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_JITRT_CUSTOM_CALLS_H_

#include <cstdint>
#include <memory>
#include <tuple>

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/type_id.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/conv.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/tracing.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"

namespace xla {
namespace gpu {
class JitRtKernelsCache;
class JitRtGemmConfigCache;
class JitRtCollectiveSupport;
class JitRtAsyncCollectiveSupport;

// Populate custom calls implementing XLA GPU runtime API.
void PopulateXlaGpuCustomCalls(runtime::DirectCustomCallRegistry& registry);

// Populate mapping from XLA (SE) enums/structs type id to symbol names.
void PopulateXlaGpuTypeIdNames(runtime::TypeIDNameRegistry& registry);

// Populate encoding from LMHLO attributes to XLA(SE) enums and structs.
void PopulateLmhloToXlaAttrEncoding(
    runtime::CustomCallAttrEncodingSet& encoding);

class JitRtGemmConfigCache {
 public:
  const GemmConfig* Get(int64_t uid);
  const GemmConfig* Set(int64_t uid, GemmConfig config);

 private:
  mutable absl::Mutex mutex_;

  llvm::SmallDenseMap<int64_t, GemmConfig> configs_ ABSL_GUARDED_BY(mutex_);
};

class JitRtCollectiveSupport {
 public:
  // Maybe block host after the first call to the collective operation with the
  // given uid, to ensure that all devices have allocated the required buffers
  // for their communicators before allowing any device to continue enqueuing
  // operations. Otherwise, the allocations can cause deadlock in the CUDA
  // driver.
  //
  // This basically ports workaround form cr/435058849 to JitRt (see details in
  // the b/215649390).
  Status MaybeBlockAfterFirstRun(int32_t uid, int32_t device_ordinal,
                                 se::Stream* stream);

 private:
  static int64_t Key(int32_t uid, int32_t device_ordinal) {
    return static_cast<int64_t>(uid) << 32 | device_ordinal;
  }

  mutable absl::Mutex mutex_;

  // Store if a particular collective operation was executed at least once. We
  // rely on unique `uid` assigned to each collective operation by the lowering
  // pass.
  llvm::SmallDenseMap<int64_t, bool> executed_ ABSL_GUARDED_BY(mutex_);
};

// Support for running async collective operations communicating via events.
class JitRtAsyncCollectiveSupport {
 public:
  explicit JitRtAsyncCollectiveSupport(se::Stream* async_comm_stream);

  mlir::FailureOr<se::Event> PopEvent(int32_t uid, int32_t device_ordinal);
  mlir::LogicalResult PushEvent(int32_t uid, int32_t device_ordinal,
                                se::Event done_event);

  ::stream_executor::Stream* async_comm_stream() const {
    return async_comm_stream_;
  }

 private:
  static int64_t EventKey(int32_t uid, int32_t device_ordinal) {
    return static_cast<int64_t>(uid) << 32 | device_ordinal;
  }

  mutable absl::Mutex mutex_;

  ::stream_executor::Stream* async_comm_stream_;

  // Store done events for the AllReduceDone to wait on.
  llvm::SmallDenseMap<int64_t, se::Event> done_events_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_JITRT_CUSTOM_CALLS_H_
