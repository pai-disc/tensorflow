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

#include "tensorflow/compiler/xla/service/gpu/jitrt_custom_calls.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/runtime/arguments.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/runtime/jit_executable.h"
#include "tensorflow/compiler/xla/runtime/type_id.h"
#include "tensorflow/compiler/xla/runtime/types.h"
#include "tensorflow/compiler/xla/service/custom_call_status_internal.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/gpu/fft_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_asm_opts_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_runner.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_gather_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/conv.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/cublas_lt_matmul.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/fft.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/io_feed.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/kernel_launch.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/tsl/platform/human_readable_json.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/service/gpu/cholesky_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/triangular_solve_thunk.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/service/gpu/runtime/graph_launch.h"
#endif  // GOOGLE_CUDA

namespace xla {
namespace gpu {

using Eigen::bfloat16;
using Eigen::half;

using llvm::ArrayRef;

using mlir::failure;
using mlir::FailureOr;
using mlir::LogicalResult;
using mlir::StringRef;
using mlir::succeeded;
using mlir::success;

using ::xla::runtime::AggregateAttrDef;
using ::xla::runtime::AggregateAttrEncoding;
using ::xla::runtime::CustomCall;
using ::xla::runtime::CustomCallAttrEncodingSet;
using ::xla::runtime::EnumAttrEncoding;
using ::xla::runtime::Executable;
using ::xla::runtime::FlatMemrefView;
using ::xla::runtime::StridedMemrefView;
using ::xla::runtime::Tagged;
using ::xla::runtime::TypeIDNameRegistry;

namespace se = ::stream_executor;
namespace lmhlo_gpu = ::mlir::lmhlo_gpu;
namespace mhlo = ::mlir::mhlo;

// Disable all CustomCall checks in optimized build.
static constexpr CustomCall::RuntimeChecks RuntimeChecks() {
#if defined(NDEBUG)
  return CustomCall::RuntimeChecks::kNone;
#else
  return CustomCall::RuntimeChecks::kDefault;
#endif
}

// -------------------------------------------------------------------------- //

// Add custom call arguments and attributes encoding for custom HLO enums and
// structs, so that we can pass them to custom calls.
void PopulateLmhloToXlaAttrEncoding(CustomCallAttrEncodingSet& encoding) {
  encoding
      .Add<EnumAttrEncoding<lmhlo_gpu::ActivationAttr, lmhlo_gpu::Activation,
                            se::dnn::ActivationMode>>(
          [](lmhlo_gpu::Activation value) -> se::dnn::ActivationMode {
            return ConvertConvActivationMode(value).value();
          });

#if GOOGLE_CUDA
  encoding.Add<EnumAttrEncoding<lmhlo_gpu::CublasLtMatmulEpilogueAttr,
                                lmhlo_gpu::CublasLtMatmulEpilogue,
                                se::cuda::BlasLt::Epilogue>>(
      [](lmhlo_gpu::CublasLtMatmulEpilogue value)
          -> se::cuda::BlasLt::Epilogue {
        return cublas_lt::AsBlasLtEpilogue(value).value();
      });
#endif  // GOOGLE_CUDA

  encoding
      .Add<EnumAttrEncoding<mhlo::FftTypeAttr, mhlo::FftType, se::fft::Type>>(
          [](mhlo::FftType value) -> se::fft::Type {
            switch (value) {
              case mhlo::FftType::FFT:
                return se::fft::Type::kC2CForward;
              case mhlo::FftType::IFFT:
                return se::fft::Type::kC2CInverse;
              case mhlo::FftType::RFFT:
                return se::fft::Type::kR2C;
              case mhlo::FftType::IRFFT:
                return se::fft::Type::kC2R;
              default:
                return se::fft::Type::kInvalid;
            }
          });

  using DotDimsAttr = mhlo::DotDimensionNumbersAttr;
  encoding.Add<
      xla::runtime::AggregateAttrEncoding<DotDimsAttr, DotDimensionNumbers>>(
      encoding,
      xla::runtime::AggregateAttrDef<DotDimsAttr>()
          .Add("lhs_batch", &DotDimsAttr::getLhsBatchingDimensions)
          .Add("lhs_contract", &DotDimsAttr::getLhsContractingDimensions)
          .Add("rhs_batch", &DotDimsAttr::getRhsBatchingDimensions)
          .Add("rhs_contract", &DotDimsAttr::getRhsContractingDimensions));

  using ConvDimsAttr = mhlo::ConvDimensionNumbersAttr;
  encoding.Add<
      xla::runtime::AggregateAttrEncoding<ConvDimsAttr, ConvDimensionNumbers>>(
      encoding,
      xla::runtime::AggregateAttrDef<ConvDimsAttr>()
          .Add("input_batch_dim", &ConvDimsAttr::getInputBatchDimension)
          .Add("input_feature_dim", &ConvDimsAttr::getInputFeatureDimension)
          .Add("input_spatial_dims", &ConvDimsAttr::getInputSpatialDimensions)
          .Add("kernel_in_feature_dim",
               &ConvDimsAttr::getKernelInputFeatureDimension)
          .Add("kernel_out_feature_dim",
               &ConvDimsAttr::getKernelOutputFeatureDimension)
          .Add("kernel_spatial_dims", &ConvDimsAttr::getKernelSpatialDimensions)
          .Add("output_batch_dim", &ConvDimsAttr::getOutputBatchDimension)
          .Add("output_feature_dim", &ConvDimsAttr::getOutputFeatureDimension)
          .Add("output_spatial_dims",
               &ConvDimsAttr::getOutputSpatialDimensions));

  using ConvConfigAttr = lmhlo_gpu::ConvolutionBackendConfigAttr;
  encoding.Add<
      xla::runtime::AggregateAttrEncoding<ConvConfigAttr, ConvBackendConfig>>(
      encoding,
      xla::runtime::AggregateAttrDef<ConvConfigAttr>()
          .Add("algorithm", &ConvConfigAttr::getAlgorithm)
          .Add("tensor_ops_enabled", &ConvConfigAttr::getTensorOpsEnabled)
          .Add("is_cudnn_frontend", &ConvConfigAttr::getIsCudnnFrontend)
          .Add("knob_ids", &ConvConfigAttr::getKnobIds)
          .Add("knob_values", &ConvConfigAttr::getKnobValues)
          .Add("operand_0_layout", &ConvConfigAttr::getOperand_0Layout)
          .Add("operand_1_layout", &ConvConfigAttr::getOperand_1Layout)
          .Add("result_layout", &ConvConfigAttr::getResultLayout)
          .Add("workspace_size", &ConvConfigAttr::getWorkspaceSize));
}

// -------------------------------------------------------------------------- //

template <typename MemrefArg>
static se::DeviceMemoryBase GetDeviceAddress(MemrefArg& memref) {
  uint64_t size = primitive_util::ByteWidth(memref.dtype);
  for (auto dim : memref.sizes) size *= dim;
  return se::DeviceMemoryBase(memref.data, size);
}

static se::DeviceMemoryBase GetDeviceAddress(runtime::FlatMemrefView& memref) {
  return se::DeviceMemoryBase(memref.data, memref.size_in_bytes);
}

// -------------------------------------------------------------------------- //

const GemmConfig* JitRtGemmConfigCache::Get(int64_t uid) {
  absl::MutexLock lock(&mutex_);
  auto it = configs_.find(uid);
  if (it != configs_.end()) return &it->second;
  return nullptr;
}

const GemmConfig* JitRtGemmConfigCache::Set(int64_t uid, GemmConfig config) {
  absl::MutexLock lock(&mutex_);
  auto it = configs_.find(uid);
  if (it != configs_.end()) return &it->second;

  auto emplaced = configs_.try_emplace(uid, std::move(config));
  return &emplaced.first->second;
}

// -------------------------------------------------------------------------- //

JitRtAsyncCollectiveSupport::JitRtAsyncCollectiveSupport(
    se::Stream* async_comm_stream)
    : async_comm_stream_(async_comm_stream) {}

Status JitRtCollectiveSupport::MaybeBlockAfterFirstRun(int32_t uid,
                                                       int32_t device_ordinal,
                                                       se::Stream* stream) {
  bool block = [&] {
    absl::MutexLock lock(&mutex_);
    return executed_.try_emplace(Key(uid, device_ordinal), true).second;
  }();
  return block ? stream->BlockHostUntilDone() : OkStatus();
}

FailureOr<se::Event> JitRtAsyncCollectiveSupport::PopEvent(
    int32_t uid, int32_t device_ordinal) {
  const int64_t key = EventKey(uid, device_ordinal);

  absl::MutexLock lock(&mutex_);
  auto it = done_events_.find(key);
  if (it == done_events_.end()) return failure();

  se::Event done_event = std::move(it->second);
  done_events_.erase(it);
  return done_event;
}

LogicalResult JitRtAsyncCollectiveSupport::PushEvent(int32_t uid,
                                                     int32_t device_ordinal,
                                                     se::Event done_event) {
  const int64_t key = EventKey(uid, device_ordinal);

  absl::MutexLock lock(&mutex_);
  auto result = done_events_.try_emplace(key, std::move(done_event));
  if (!result.second) return failure();  // done event has not been consumed

  return success();
}

// -------------------------------------------------------------------------- //

#if XLA_ENABLE_XCCL
FailureOr<NcclComm::Lock> GetNcclComm(const NcclExecuteParams& params,
                                      int64_t group_mode, int64_t op_id,
                                      ArrayRef<int64_t> replica_group_offsets,
                                      ArrayRef<int64_t> replica_group_values) {
  // TODO(b/233930690): Pass the attribute below as a nested array.
  // Pass an array of arrays using two vectors; one specifying all the values
  // and another specifying the (ending) offsets of each array in the other
  // vector. Example: [ [10, 20, 30, 40], [50, 60], [70, 80, 90] ] turns into
  // offsets=[4, 6, 9] values=[10, 20, 30, 40, 50, 60, 70, 80, 90].
  std::vector<ReplicaGroup> replica_groups;
  int i = 0;
  for (int64_t replica_group_end : replica_group_offsets) {
    ReplicaGroup replica_group;
    while (i < replica_group_end)
      replica_group.add_replica_ids(replica_group_values[i++]);
    replica_groups.push_back(replica_group);
  }

  auto comm =
      LockNcclComm(params, replica_groups,
                   static_cast<CollectiveOpGroupMode>(group_mode), op_id);
  if (comm.ok()) return std::move(comm.value());
  return failure();
}
#endif  // XLA_ENABLE_XCCL

FailureOr<std::vector<DeviceBufferPair>> GetDeviceBufferPairs(
    CustomCall::RemainingArgs& args) {
  // Add MemRef arguments as buffer arguments.
  const int buffer_pairs = args.size() / 2;
  std::vector<DeviceBufferPair> device_buffers;
  device_buffers.reserve(buffer_pairs);
  for (int i = 0; i < buffer_pairs; ++i) {
    auto source = args.get<runtime::StridedMemrefView>(i);
    auto destination = args.get<runtime::StridedMemrefView>(i + buffer_pairs);
    if (failed(source) || failed(destination)) {
      // Unsupported argument type.
      return failure();
    }

    int element_count = 1;
    for (int size : source->sizes) element_count *= size;
    device_buffers.emplace_back(DeviceBufferPair{
        source->dtype, element_count, GetDeviceAddress(*source),
        GetDeviceAddress(*destination)});
  }
  return device_buffers;
}

// -------------------------------------------------------------------------- //

namespace {
struct Gemm {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          const DebugOptions* debug_options,
                          JitRtGemmConfigCache* configs,
                          runtime::StridedMemrefView lhs,
                          runtime::StridedMemrefView rhs,
                          runtime::StridedMemrefView out, int64_t algorithm,
                          double alpha_real, double alpha_imag, double beta,
                          DotDimensionNumbers dot_dims, int64_t uid) const;

  static Gemm Handler() { return Gemm(); }
};
}  // namespace

absl::Status Gemm::operator()(const ServiceExecutableRunOptions* run_options,
                              const DebugOptions* debug_options,
                              JitRtGemmConfigCache* configs,
                              runtime::StridedMemrefView lhs,
                              runtime::StridedMemrefView rhs,
                              runtime::StridedMemrefView out, int64_t algorithm,
                              double alpha_real, double alpha_imag, double beta,
                              DotDimensionNumbers dot_dims, int64_t uid) const {
  se::DeviceMemoryBase lhs_data = GetDeviceAddress(lhs);
  se::DeviceMemoryBase rhs_data = GetDeviceAddress(rhs);
  se::DeviceMemoryBase output_data = GetDeviceAddress(out);

  VLOG(3) << "Running GEMM";
  se::Stream* stream = run_options->stream();

  // Find the gemm config for this instance of operation based on uid.
  const GemmConfig* config = configs->Get(uid);
  if (config == nullptr) {
    auto cfg = GetGemmConfig(lhs, rhs, out, algorithm, alpha_real, alpha_imag,
                             beta, dot_dims.lhs_batch, dot_dims.lhs_contract,
                             dot_dims.rhs_batch, dot_dims.rhs_contract);
    if (!cfg.ok()) return ToAbslStatus(cfg.status());
    config = configs->Set(uid, std::move(*cfg));
  }

  Status executed = [&]() -> Status {
    return RunGemm(*config, lhs_data, rhs_data, output_data, stream);
  }();

  if (!executed.ok()) return ToAbslStatus(executed);

  return absl::OkStatus();
}

static bool Gemm(runtime::ExecutionContext* ctx, void** args, void** attrs,
                 void** rets) {
  static auto* handler = CustomCall::Bind("xla.gpu.gemm")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .UserData<const DebugOptions*>()
                             .UserData<JitRtGemmConfigCache*>()
                             .Arg<runtime::StridedMemrefView>()  // lhs
                             .Arg<runtime::StridedMemrefView>()  // rhs
                             .Arg<runtime::StridedMemrefView>()  // out
                             .Attr<int64_t>("algorithm")
                             .Attr<double>("alpha_real")
                             .Attr<double>("alpha_imag")
                             .Attr<double>("beta")
                             .Attr<DotDimensionNumbers>("dot_dims")
                             .Attr<int64_t>("uid")
                             .To<RuntimeChecks()>(Gemm::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {

enum class MemcpyDirection { kDeviceToDevice, kDeviceToHost, kHostToDevice };

template <MemcpyDirection direction>
struct Memcpy {
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          runtime::StridedMemrefView dst,
                          runtime::StridedMemrefView src) const;
  static Memcpy Handler() { return Memcpy(); }
};
}  // namespace

template <MemcpyDirection direction>
absl::Status Memcpy<direction>::operator()(
    const ServiceExecutableRunOptions* run_options,
    runtime::StridedMemrefView dst, runtime::StridedMemrefView src) const {
  se::Stream* stream = run_options->stream();

  if (dst.sizes != src.sizes) {
    return absl::InvalidArgumentError(
        "Source memref sizes do not match destination memref sizes");
  }

  if (dst.strides != src.strides) {
    return absl::InvalidArgumentError(
        "Source memref strides do not match destination memref strides");
  }

  switch (direction) {
    case MemcpyDirection::kDeviceToDevice: {
      se::DeviceMemoryBase dst_data = GetDeviceAddress(dst);
      se::DeviceMemoryBase src_data = GetDeviceAddress(src);
      stream->ThenMemcpy(&dst_data, src_data, src_data.size());
    } break;
    case MemcpyDirection::kDeviceToHost: {
      se::DeviceMemoryBase src_data = GetDeviceAddress(src);
      stream->ThenMemcpy(dst.data, src_data, src_data.size());
    } break;
    case MemcpyDirection::kHostToDevice: {
      se::DeviceMemoryBase dst_data = GetDeviceAddress(dst);
      stream->ThenMemcpy(&dst_data, src.data, dst_data.size());
    } break;
  }

  // TODO(ezhulenev): H2D and D2H memcpy instead of blocking the execution
  // thread should return an async token that will become available when
  // transfer is completed.
  if (direction != MemcpyDirection::kDeviceToDevice) {
    auto st = stream->BlockHostUntilDone();
    if (!st.ok()) return ToAbslStatus(st);
  }

  return absl::OkStatus();
}

template <MemcpyDirection direction>
static bool MemcpyFn(runtime::ExecutionContext* ctx, void** args, void** attrs,
                     void** rets) {
  static auto* handler = CustomCall::Bind("xla.gpu.memcpy")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .Arg<runtime::StridedMemrefView>()  // dst
                             .Arg<runtime::StridedMemrefView>()  // src
                             .To<RuntimeChecks()>(Memcpy<direction>::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {

struct Memset {
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          runtime::StridedMemrefView dst,
                          CustomCall::VariantArg constant) const;
  static Memset Handler() { return Memset(); }
};

}  // namespace

// TODO(ezhulenev): Add `VariantArg` type dispatching for all scalar types
// supported by Xla (PrimitiveType).

// Checks all supported data types to see if the value is zero.
static bool IsZero(CustomCall::VariantArg constant) {
  if (auto i1 = constant.get<bool>(); succeeded(i1))
    return *i1 == false;
  else if (auto i8 = constant.get<int8_t>(); succeeded(i8))
    return *i8 == 0;
  else if (auto i16 = constant.get<int16_t>(); succeeded(i16))
    return *i16 == 0;
  else if (auto i32 = constant.get<int32_t>(); succeeded(i32))
    return *i32 == 0;
  else if (auto i64 = constant.get<int64_t>(); succeeded(i64))
    return *i64 == 0;
  else if (auto bf16 = constant.get<bfloat16>(); succeeded(bf16))
    return *bf16 == bfloat16(0.0);
  else if (auto f16 = constant.get<half>(); succeeded(f16))
    return *f16 == half(0.0);
  else if (auto f32 = constant.get<float>(); succeeded(f32))
    return *f32 == 0.0;
  else if (auto f64 = constant.get<double>(); succeeded(f64))
    return *f64 == 0.0;

  return false;
}

// Convert constant value to 32-bit pattern.
static absl::StatusOr<uint32_t> ToBitPattern(CustomCall::VariantArg constant) {
  // If the value is 8 or 16 bits wide, we can emit a 32-bit memset by
  // repeating the value 4 or 2 times, so long as the destination buffer is
  // an even multiple of 32 bits long.
  //
  // This code is identical to `ir_emitter_unnested`.
  //
  // We use `memcpy` operation to copy bytes between value and the uint32_t bit
  // pattern because in theory they might have incompatible alignment, and we
  // rely on LLVM to optimize it.
  auto extend = [](auto value) -> uint32_t {
    static constexpr size_t num_bytes = sizeof(value);
    static_assert(num_bytes < 4);

    uint16_t pattern16;
    if constexpr (num_bytes == 1) {
      uint8_t b = value;
      pattern16 = uint16_t{b} | (uint16_t{b} << 8);
    } else {
      memcpy(&pattern16, &value, sizeof(pattern16));
    }
    return uint32_t{pattern16} | (uint32_t{pattern16} << 16);
  };

  // Truncate value to 32-bit pattern.
  auto truncate = [](auto value) -> uint32_t {
    static_assert(sizeof(value) >= 4);

    uint32_t pattern;
    memcpy(&pattern, &value, sizeof(pattern));
    return pattern;
  };

  if (auto i1 = constant.get<bool>(); succeeded(i1))
    return extend(*i1);
  else if (auto i8 = constant.get<int8_t>(); succeeded(i8))
    return extend(*i8);
  else if (auto i16 = constant.get<int16_t>(); succeeded(i16))
    return extend(*i16);
  else if (auto i32 = constant.get<int32_t>(); succeeded(i32))
    return truncate(*i32);
  else if (auto i64 = constant.get<int64_t>(); succeeded(i64))
    return truncate(*i64);
  else if (auto bf16 = constant.get<bfloat16>(); succeeded(bf16))
    return extend(static_cast<uint16_t>(*bf16));
  else if (auto f16 = constant.get<half>(); succeeded(f16))
    return extend(static_cast<uint16_t>(*f16));
  else if (auto f32 = constant.get<float>(); succeeded(f32))
    return truncate(*f32);
  else if (auto f64 = constant.get<double>(); succeeded(f64))
    return truncate(*f64);

  return absl::InvalidArgumentError("Unsupported memset constant type");
}

absl::Status Memset::operator()(const ServiceExecutableRunOptions* run_options,
                                runtime::StridedMemrefView dst,
                                CustomCall::VariantArg constant) const {
  se::Stream* stream = run_options->stream();
  se::DeviceMemoryBase dst_data = GetDeviceAddress(dst);

  // If the constant is zero we can use memzero directly.
  if (IsZero(constant)) {
    stream->ThenMemZero(&dst_data, dst_data.size());
    return absl::OkStatus();
  }

  // If the constant is not zero, use the given pattern to `memset`.
  absl::StatusOr<uint32_t> pattern = ToBitPattern(constant);
  if (!pattern.ok()) return pattern.status();

  if (dst_data.size() % 4 != 0)
    return absl::InvalidArgumentError("Memref size is not divisible by 4");

  stream->ThenMemset32(&dst_data, *pattern, dst_data.size());

  return absl::OkStatus();
}

static bool MemsetFn(runtime::ExecutionContext* ctx, void** args, void** attrs,
                     void** rets) {
  static auto* handler = CustomCall::Bind("xla.gpu.memset")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .Arg<runtime::StridedMemrefView>()  // dst
                             .Arg<CustomCall::VariantArg>()      // constant
                             .To<RuntimeChecks()>(Memset::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {
struct Cholesky {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          const DebugOptions* debug_options,
                          runtime::StridedMemrefView operand,
                          runtime::StridedMemrefView a,
                          runtime::MemrefView workspace,
                          runtime::MemrefView info, int64_t batch_size,
                          bool is_lower, int64_t n) const;
  static Cholesky Handler() { return Cholesky(); }
};
}  // namespace

absl::Status Cholesky::operator()(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, runtime::StridedMemrefView operand,
    runtime::StridedMemrefView a, runtime::MemrefView workspace,
    runtime::MemrefView info, int64_t batch_size, bool is_lower,
    int64_t n) const {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  se::DeviceMemoryBase operand_buffer = GetDeviceAddress(operand);
  se::DeviceMemoryBase a_buffer = GetDeviceAddress(a);
  se::DeviceMemoryBase workspace_buffer = GetDeviceAddress(workspace);
  se::DeviceMemoryBase info_buffer = GetDeviceAddress(info);

  VLOG(3) << "Running Cholesky";
  se::Stream* stream = run_options->stream();

  // Copy operand to the a buffer if they are different.
  if (a.data != operand.data)
    stream->ThenMemcpy(&a_buffer, operand_buffer, operand_buffer.size());

  using UpperLower = se::blas::UpperLower;
  UpperLower uplo = is_lower ? UpperLower::kLower : UpperLower::kUpper;

  CholeskyParams params{n,        batch_size,       uplo,
                        a_buffer, workspace_buffer, info_buffer};
  auto executed = RunCholesky(xla::gpu::PtxOptsFromDebugOptions(*debug_options),
                              operand.dtype, &params, stream);
  if (!executed.ok()) return ToAbslStatus(executed);

  return absl::OkStatus();
#else  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return absl::InternalError("Not implemented without Gpu");
#endif
}

static bool Cholesky(runtime::ExecutionContext* ctx, void** args, void** attrs,
                     void** rets) {
  static auto* handler = CustomCall::Bind("xla.gpu.cholesky")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .UserData<const DebugOptions*>()
                             .Arg<runtime::StridedMemrefView>()  // operand
                             .Arg<runtime::StridedMemrefView>()  // a
                             .Arg<runtime::MemrefView>()         // workspace
                             .Arg<runtime::MemrefView>()         // info
                             .Attr<int64_t>("batch_size")
                             .Attr<bool>("is_lower")
                             .Attr<int64_t>("n")
                             .To<RuntimeChecks()>(Cholesky::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {

// TODO(ezhulenev): Today XLA represents TriangularSolve as a "classic" XLA
// custom call operation, and we provide a thin adaptor from Xla custom call
// to JitRt custom call. Once we are fully migrated to JitRt exectuion, XLA
// compiler should directly emit properly typed TriangularSolve JitRt custom
// call (no need to pass config via the serialized string).
struct TriangularSolve {
  // Adaptor from XlaCustomCall API to properly typed TriangularSolve handler.
  static absl::Status run(const ServiceExecutableRunOptions* run_options,
                          const DebugOptions* debug_options,
                          CustomCall::RemainingArgs args,
                          StringRef backend_config);

  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          const DebugOptions* debug_options,
                          runtime::StridedMemrefView a,
                          runtime::StridedMemrefView b,
                          runtime::StridedMemrefView result,
                          runtime::FlatMemrefView temp, bool left_side,
                          bool lower, bool unit_diagonal,
                          TriangularSolveOptions::Transpose transpose_a) const;
  static TriangularSolve Handler() { return TriangularSolve(); }
};

}  // namespace

absl::Status TriangularSolve::run(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, CustomCall::RemainingArgs args,
    StringRef backend_config) {
  TriangularSolve handler = TriangularSolve::Handler();

  if (args.size() != 4)
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected 4 arguments, got %d", args.size()));

  // Check if all arguments have the correct type.
  auto a = args.get<runtime::StridedMemrefView>(0);
  auto b = args.get<runtime::StridedMemrefView>(1);
  auto result = args.get<runtime::StridedMemrefView>(2);
  auto temp = args.get<runtime::FlatMemrefView>(3);
  if (failed(a) || failed(b) || failed(result) || failed(temp))
    return absl::InvalidArgumentError("Incorrect argument types");

  // Parse backend config string.
  TriangularSolveOptions opts;
  auto st = tsl::HumanReadableJsonToProto(backend_config.str(), &opts);
  if (!st.ok()) return ToAbslStatus(st);

  return handler(run_options, debug_options, *a, *b, *result, *temp,
                 opts.left_side(), opts.lower(), opts.unit_diagonal(),
                 opts.transpose_a());
}

absl::Status TriangularSolve::operator()(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, runtime::StridedMemrefView a,
    runtime::StridedMemrefView b, runtime::StridedMemrefView result,
    runtime::FlatMemrefView temp, bool left_side, bool lower,
    bool unit_diagonal, TriangularSolveOptions::Transpose transpose_a) const {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  se::Stream* stream = run_options->stream();

  se::DeviceMemoryBase a_data = GetDeviceAddress(a);
  se::DeviceMemoryBase b_data = GetDeviceAddress(b);
  se::DeviceMemoryBase result_data = GetDeviceAddress(result);
  se::DeviceMemoryBase temp_data = GetDeviceAddress(temp);

  // Triangular solve is in-place on 'b', so copy 'b' to the output if they
  // aren't the same buffer.
  if (b.data != result.data)
    stream->ThenMemcpy(&result_data, b_data, b_data.size());

  Shape b_shape = ToShape(b);
  int64_t m = b_shape.dimensions(b_shape.rank() - 2);
  int64_t n = b_shape.dimensions(b_shape.rank() - 1);
  int64_t batch_size = std::accumulate(
      b_shape.dimensions().begin(), b_shape.dimensions().end() - 2, int64_t{1},
      [](int64_t a, int64_t b) { return a * b; });

  PrimitiveType elem_type = b.dtype;
  int64_t elem_size = ShapeUtil::ByteSizeOfPrimitiveType(elem_type);
  int64_t a_batch_stride = left_side ? m * m * elem_size : n * n * elem_size;
  int64_t b_batch_stride = m * n * elem_size;

  using Side = se::blas::Side;
  using Diagonal = se::blas::Diagonal;
  using Transpose = se::blas::Transpose;
  using UpperLower = se::blas::UpperLower;

  // Convert custom call attributes to se::blas enums.
  UpperLower uplo = lower ? UpperLower::kLower : UpperLower::kUpper;
  Side side = left_side ? Side::kLeft : Side::kRight;
  Diagonal diagonal = unit_diagonal ? Diagonal::kUnit : Diagonal::kNonUnit;

  auto transpose = [&]() -> mlir::FailureOr<Transpose> {
    switch (transpose_a) {
      case TriangularSolveOptions::NO_TRANSPOSE:
        return se::blas::Transpose::kNoTranspose;
      case TriangularSolveOptions::TRANSPOSE:
        return se::blas::Transpose::kTranspose;
      case TriangularSolveOptions::ADJOINT:
        return se::blas::Transpose::kConjugateTranspose;
      default:
        return failure();
    }
  }();

  if (failed(transpose))
    return absl::InternalError("Failed to convert transpose type");

  auto st = RunTriangulatSolve(
      a_data, result_data, temp_data, PtxOptsFromDebugOptions(*debug_options),
      uplo, side, diagonal, *transpose, elem_type, batch_size, m, n,
      a_batch_stride, b_batch_stride, stream);
  if (!st.ok()) return ToAbslStatus(st);

  return absl::OkStatus();
#else  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return absl::InternalError("Not implemented without Gpu");
#endif
}

// -------------------------------------------------------------------------- //
// Implements JitRt custom call that forward to the Xla Custom Call handler.
//
// Longer term all Xla custom calls probably should be directly implemented as
// JitRt custom calls. However for smooth migration from Thunks to JitRt we have
// to seamlessly support all current XLA users.

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
namespace {
struct XlaCustomCall {
  using Stream = se::gpu::GpuStreamHandle;

  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          const DebugOptions* debug_options,
                          CustomCall::RemainingArgs args,
                          StringRef call_target_name, int32_t api_version,
                          StringRef backend_config) const;
  static XlaCustomCall Handler() { return XlaCustomCall(); }
};
}  // namespace

absl::Status XlaCustomCall::operator()(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, CustomCall::RemainingArgs args,
    StringRef call_target_name, int32_t api_version,
    StringRef backend_config) const {
  // Pattern match custom call to a few special cases, otherwise find the custom
  // call handler regustered with the runtime.
  if (call_target_name == kTriangularSolveCallTarget)
    return TriangularSolve::run(run_options, debug_options, args,
                                backend_config);

  // Find the Xla custom call handler.
  auto& platform_name = run_options->stream()->parent()->platform()->Name();
  void* call_target = CustomCallTargetRegistry::Global()->Lookup(
      call_target_name.str(), platform_name);
  if (!call_target) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cannot find the Xla custom call handler ", call_target_name.str()));
  }

  // Prepare pointers to buffers to pass to the Xla custom call handler.
  llvm::SmallVector<void*> buffers;
  for (unsigned i = 0; i < args.size(); ++i) {
    if (auto memref = args.get<FlatMemrefView>(i); succeeded(memref)) {
      buffers.push_back(memref->data);
      continue;
    }

    if (auto strided = args.get<StridedMemrefView>(i); succeeded(strided)) {
      int64_t size_in_bytes = primitive_util::ByteWidth(strided->dtype);
      for (int64_t size : strided->sizes) size_in_bytes *= size;
      buffers.push_back(strided->data);
      continue;
    }

    // TODO(ezhulenev): Add dialect and type to model Xla custom call holes,
    // today we rely on the fact that custom calls do not support scalar
    // arguments and we can disambiguate holes from real arguments.
    if (auto hole = args.get<int64_t>(i); succeeded(hole)) {
      buffers.push_back(nullptr);
      continue;
    }

    return absl::InvalidArgumentError(
        "Failed to get arguments as (strided) memref view");
  }

  // Original custom call API version that doesn't support returning status.
  if (api_version == CustomCallApiVersion::API_VERSION_ORIGINAL) {
    using XlaCustomCallType = void (*)(Stream, void**, const char*, size_t);
    auto xla_call_target = reinterpret_cast<XlaCustomCallType>(call_target);

    xla_call_target(se::gpu::AsGpuStreamValue(run_options->stream()),
                    buffers.data(), backend_config.data(),
                    backend_config.size());

    return absl::OkStatus();
  }

  // Xla Custom call API returning status.
  if (api_version == CustomCallApiVersion::API_VERSION_STATUS_RETURNING ||
      api_version ==
          CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED) {
    using XlaCustomCallType =
        void (*)(Stream, void**, const char*, size_t, XlaCustomCallStatus*);
    auto xla_call_target = reinterpret_cast<XlaCustomCallType>(call_target);

    XlaCustomCallStatus custom_call_status;
    xla_call_target(se::gpu::AsGpuStreamValue(run_options->stream()),
                    buffers.data(), backend_config.data(),
                    backend_config.size(), &custom_call_status);

    if (auto message = CustomCallStatusGetMessage(&custom_call_status)) {
      return absl::InternalError(message.value());
    } else {
      return absl::OkStatus();
    }
  }

  return absl::InvalidArgumentError("Incorrect custom call API version");
}

static bool CustomCall(runtime::ExecutionContext* ctx, void** args,
                       void** attrs, void** rets) {
  static auto* handler = CustomCall::Bind("xla.gpu.memcpy")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .UserData<const DebugOptions*>()
                             .Arg<CustomCall::RemainingArgs>()  // args
                             .Attr<std::string_view>("call_target_name")
                             .Attr<int32_t>("api_version")
                             .Attr<std::string_view>("backend_config")
                             .To<RuntimeChecks()>(XlaCustomCall::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// ------------------------------------------------------------------------- //

namespace {
struct AllReduce {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          JitRtCollectiveSupport* collectives,
                          CustomCall::RemainingArgs args, int32_t uid,
                          int64_t group_mode, int64_t op_id,
                          int64_t reduction_kind,
                          ArrayRef<int64_t> replica_group_offsets,
                          ArrayRef<int64_t> replica_group_values) const;
  static AllReduce Handler() { return AllReduce(); }
};
}  // namespace

absl::Status AllReduce::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id, int64_t reduction_kind,
    ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllReduce";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return absl::InternalError("Failed to get NcclComm");

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers))
    return absl::InternalError("Failed to get device buffers");

  auto executed = RunAllReduce(static_cast<ReductionKind>(reduction_kind),
                               *device_buffers, *stream, **comm);
  if (!executed.ok()) return ToAbslStatus(executed);

  int32_t device_ordinal = stream->parent()->device_ordinal();
  auto st = collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream);
  if (!st.ok()) return ToAbslStatus(st);

  return absl::OkStatus();
#else   // XLA_ENABLE_XCCL
  // NCCL disabled.
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

static bool AllReduce(runtime::ExecutionContext* ctx, void** args, void** attrs,
                      void** rets) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.all_reduce")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtCollectiveSupport*>()
          .RemainingArgs()  // args
          .Attr<int32_t>("uid")
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<int64_t>("op_id")
          .Attr<int64_t>("reduction_kind")  // ReductionKind
          .Attr<ArrayRef<int64_t>>("replica_group_offsets")
          .Attr<ArrayRef<int64_t>>("replica_group_values")
          .To<RuntimeChecks()>(AllReduce::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// ------------------------------------------------------------------------- //

namespace {
struct AllReduceStart {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          JitRtAsyncCollectiveSupport* async_collectives,
                          CustomCall::RemainingArgs args, int64_t group_mode,
                          int64_t op_id, int64_t reduction_kind,
                          ArrayRef<int64_t> replica_group_offsets,
                          ArrayRef<int64_t> replica_group_values,
                          int32_t uid) const;
  static AllReduceStart Handler() { return AllReduceStart(); }
};
}  // namespace

absl::Status AllReduceStart::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtAsyncCollectiveSupport* async_collectives,
    CustomCall::RemainingArgs args, int64_t group_mode, int64_t op_id,
    int64_t reduction_kind, ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values, int32_t uid) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllReduceStart";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return absl::InternalError("Failed to get NcclComm");

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers))
    return absl::InternalError("Failed to get device buffers");

  // Wait until compute inputs are ready.
  async_collectives->async_comm_stream()->ThenWaitFor(params.stream);

  auto executed =
      RunAllReduce(static_cast<ReductionKind>(reduction_kind), *device_buffers,
                   *async_collectives->async_comm_stream(), **comm);
  if (!executed.ok()) return ToAbslStatus(executed);

  // Create an event on the async stream for the completion of the all-reduce.
  se::Event done_event(async_collectives->async_comm_stream()->parent());
  if (!done_event.Init()) return absl::InternalError("Failed to create event");
  async_collectives->async_comm_stream()->ThenRecordEvent(&done_event);

  if (failed(async_collectives->PushEvent(
          uid, stream->parent()->device_ordinal(), std::move(done_event))))
    return absl::InternalError("Failed to push event to async collectives");

  return absl::OkStatus();
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

static bool AllReduceStart(runtime::ExecutionContext* ctx, void** args,
                           void** attrs, void** rets) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.all_reduce_start")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtAsyncCollectiveSupport*>()
          .RemainingArgs()              // args
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<int64_t>("op_id")
          .Attr<int64_t>("reduction_kind")  // ReductionKind
          .Attr<ArrayRef<int64_t>>("replica_group_offsets")
          .Attr<ArrayRef<int64_t>>("replica_group_values")
          .Attr<int32_t>("uid")
          .To<RuntimeChecks()>(AllReduceStart::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// ------------------------------------------------------------------------- //

namespace {
struct AllReduceDone {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          JitRtCollectiveSupport* collectives,
                          JitRtAsyncCollectiveSupport* async_collectives,
                          CustomCall::RemainingArgs args, int32_t uid) const;
  static AllReduceDone Handler() { return AllReduceDone(); }
};
}  // namespace

absl::Status AllReduceDone::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives,
    JitRtAsyncCollectiveSupport* async_collectives,
    CustomCall::RemainingArgs args, int32_t uid) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllReduceDone";
  se::Stream* stream = run_options->stream();

  int32_t device_ordinal = stream->parent()->device_ordinal();
  auto event = async_collectives->PopEvent(uid, device_ordinal);
  if (failed(event)) return absl::InternalError("Failed to pop event");

  stream->ThenWaitFor(&*event);

  if (!collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream).ok())
    return absl::InternalError("Failed to block host");

  return absl::OkStatus();
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

static bool AllReduceDone(runtime::ExecutionContext* ctx, void** args,
                          void** attrs, void** rets) {
  static auto* handler = CustomCall::Bind("xla.gpu.all_reduce_done")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .UserData<JitRtCollectiveSupport*>()
                             .UserData<JitRtAsyncCollectiveSupport*>()
                             .RemainingArgs()  // args
                             .Attr<int32_t>("uid")
                             .To<RuntimeChecks()>(AllReduceDone::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {
struct ReduceScatter {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          JitRtCollectiveSupport* collectives,
                          CustomCall::RemainingArgs args, int32_t uid,
                          int64_t group_mode, int64_t op_id,
                          int64_t reduction_kind,
                          ArrayRef<int64_t> replica_group_offsets,
                          ArrayRef<int64_t> replica_group_values) const;
  static ReduceScatter Handler() { return ReduceScatter(); }
};
}  // namespace

absl::Status ReduceScatter::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id, int64_t reduction_kind,
    ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running ReduceScatter";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return absl::InternalError("Failed to get NcclComm");

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers))
    return absl::InternalError("Failed to get device buffers");

  auto executed = RunReduceScatter(static_cast<ReductionKind>(reduction_kind),
                                   *device_buffers, *stream, **comm);
  if (!executed.ok()) return ToAbslStatus(executed);

  int32_t device_ordinal = stream->parent()->device_ordinal();
  if (!collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream).ok())
    return absl::InternalError("Failed to block host");

  return absl::OkStatus();
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

static bool ReduceScatter(runtime::ExecutionContext* ctx, void** args,
                          void** attrs, void** rets) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.reduce_scatter")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtCollectiveSupport*>()
          .RemainingArgs()  // args
          .Attr<int32_t>("uid")
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<int64_t>("op_id")
          .Attr<int64_t>("reduction_kind")  // ReductionKind
          .Attr<ArrayRef<int64_t>>("replica_group_offsets")
          .Attr<ArrayRef<int64_t>>("replica_group_values")
          .To<RuntimeChecks()>(ReduceScatter::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {
struct AllGather {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          JitRtCollectiveSupport* collectives,
                          CustomCall::RemainingArgs args, int32_t uid,
                          int64_t group_mode, int64_t op_id,
                          ArrayRef<int64_t> replica_group_offsets,
                          ArrayRef<int64_t> replica_group_values) const;
  static AllGather Handler() { return AllGather(); }
};
}  // namespace

absl::Status AllGather::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id,
    ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllGather";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return absl::InternalError("Failed to get NCCL comm");

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers))
    return absl::InternalError("Failed to get device buffers");

  auto st = RunAllGather(*device_buffers, *stream, **comm);
  if (!st.ok()) return ToAbslStatus(st);

  int32_t device_ordinal = stream->parent()->device_ordinal();
  st = collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream);
  if (!st.ok()) return ToAbslStatus(st);

  return absl::OkStatus();
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL diasbled");
#endif  // XLA_ENABLE_XCCL
}

static bool AllGather(runtime::ExecutionContext* ctx, void** args, void** attrs,
                      void** rets) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.all_gather")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtCollectiveSupport*>()
          .RemainingArgs()  // args
          .Attr<int32_t>("uid")
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<int64_t>("op_id")
          .Attr<ArrayRef<int64_t>>("replica_group_offsets")
          .Attr<ArrayRef<int64_t>>("replica_group_values")
          .To<RuntimeChecks()>(AllGather::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {
struct AllToAll {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          JitRtCollectiveSupport* collectives,
                          CustomCall::RemainingArgs args, int32_t uid,
                          int64_t group_mode, bool has_split_dimension,
                          int64_t op_id,
                          ArrayRef<int64_t> replica_group_offsets,
                          ArrayRef<int64_t> replica_group_values) const;
  static AllToAll Handler() { return AllToAll(); }
};
}  // namespace

absl::Status AllToAll::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, bool has_split_dimension, int64_t op_id,
    ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllToAll";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return absl::InternalError("Failed to get NCCL comm");

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers))
    return absl::InternalError("Failed to get device buffers");

  auto st = RunAllToAll(has_split_dimension, *device_buffers, *stream, **comm);
  if (!st.ok()) return ToAbslStatus(st);

  int32_t device_ordinal = stream->parent()->device_ordinal();
  st = collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream);
  if (!st.ok()) return ToAbslStatus(st);

  return absl::OkStatus();
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

static bool AllToAll(runtime::ExecutionContext* ctx, void** args, void** attrs,
                     void** rets) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.all_to_all")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtCollectiveSupport*>()
          .RemainingArgs()  // args
          .Attr<int32_t>("uid")
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<bool>("has_split_dimension")
          .Attr<int64_t>("op_id")
          .Attr<ArrayRef<int64_t>>("replica_group_offsets")
          .Attr<ArrayRef<int64_t>>("replica_group_values")
          .To<RuntimeChecks()>(AllToAll::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {
struct CollectivePermute {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          JitRtCollectiveSupport* collectives,
                          CustomCall::RemainingArgs args, int32_t uid,
                          int64_t group_mode, int64_t op_id,
                          ArrayRef<int64_t> replica_group_offsets,
                          ArrayRef<int64_t> replica_group_values,
                          ArrayRef<int64_t> source_peers,
                          ArrayRef<int64_t> target_peers) const;
  static CollectivePermute Handler() { return CollectivePermute(); }
};
}  // namespace

absl::Status CollectivePermute::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id,
    ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values, ArrayRef<int64_t> source_peers,
    ArrayRef<int64_t> target_peers) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running CollectivePermute";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return absl::InternalError("Failed to get NcclComm");

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers))
    return absl::InternalError("Failed to get device buffers");
  if (device_buffers->size() != 1) {
    return absl::InternalError(absl::StrFormat(
        "Expected device buffer size: 1, got %d", device_buffers->size()));
  }

  StatusOr<GlobalDeviceId> global_device_id = params.GetGlobalDeviceId();
  if (!global_device_id.ok()) return ToAbslStatus(global_device_id.status());

  StatusOr<DeviceAssignment::LogicalID> current_logical_id =
      params.device_assn->LogicalIdForDevice(global_device_id.value());
  if (!current_logical_id.ok())
    return ToAbslStatus(current_logical_id.status());

  const int64_t current_id = static_cast<CollectiveOpGroupMode>(group_mode) ==
                                     CollectiveOpGroupMode::kCrossReplica
                                 ? current_logical_id.value().replica_id
                                 : current_logical_id.value().computation_id;
  std::string device_string = NcclCollectiveThunk::GetDeviceString(params);

  NcclCollectivePermuteConfig::IdToSourceTargetMap id_to_source_target;
  for (int i = 0; i < source_peers.size(); ++i) {
    id_to_source_target.insert({target_peers[i], {}}).first->second.source =
        source_peers[i];
    id_to_source_target.insert({source_peers[i], {}}).first->second.target =
        target_peers[i];
  }
  const NcclCollectivePermuteConfig::SourceTargetMapEntry source_target =
      NcclCollectivePermuteConfig::GetSourceTarget(id_to_source_target,
                                                   current_id);

  auto executed =
      RunCollectivePermute(source_target, (*device_buffers)[0], *stream, **comm,
                           device_string, current_id);
  if (!executed.ok()) return ToAbslStatus(executed);

  int32_t device_ordinal = stream->parent()->device_ordinal();
  auto st = collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream);
  if (!st.ok()) return ToAbslStatus(st);

  return absl::OkStatus();
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

static bool CollectivePermute(runtime::ExecutionContext* ctx, void** args,
                              void** attrs, void** rets) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.collective_permute")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtCollectiveSupport*>()
          .RemainingArgs()  // args
          .Attr<int32_t>("uid")
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<int64_t>("op_id")
          .Attr<ArrayRef<int64_t>>("replica_group_offsets")
          .Attr<ArrayRef<int64_t>>("replica_group_values")
          .Attr<ArrayRef<int64_t>>("source_peers")
          .Attr<ArrayRef<int64_t>>("target_peers")
          .To<RuntimeChecks()>(CollectivePermute::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {
struct ReplicaId {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          runtime::FlatMemrefView result) const;
  static ReplicaId Handler() { return ReplicaId(); }
};
}  // namespace

absl::Status ReplicaId::operator()(
    const ServiceExecutableRunOptions* run_options,
    runtime::FlatMemrefView result) const {
  VLOG(3) << "Running ReplicaId";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  StatusOr<GlobalDeviceId> global_device_id = params.GetGlobalDeviceId();
  if (!global_device_id.ok()) return ToAbslStatus(global_device_id.status());

  StatusOr<DeviceAssignment::LogicalID> logical_id =
      params.device_assn->LogicalIdForDevice(global_device_id.value());
  if (!logical_id.ok()) return ToAbslStatus(logical_id.status());

  se::DeviceMemoryBase result_data = GetDeviceAddress(result);
  params.stream->ThenMemset32(&result_data, logical_id.value().replica_id,
                              /*size=*/4);

  return absl::OkStatus();
}

static bool ReplicaId(runtime::ExecutionContext* ctx, void** args, void** attrs,
                      void** rets) {
  static auto* handler = CustomCall::Bind("xla.gpu.replica_id")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .Arg<runtime::FlatMemrefView>()  // result
                             .To<RuntimeChecks()>(ReplicaId::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {
struct PartitionId {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          runtime::FlatMemrefView result) const;
  static PartitionId Handler() { return PartitionId(); }
};
}  // namespace

absl::Status PartitionId::operator()(
    const ServiceExecutableRunOptions* run_options,
    runtime::FlatMemrefView result) const {
  VLOG(3) << "Running PartitionId";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  StatusOr<GlobalDeviceId> global_device_id = params.GetGlobalDeviceId();
  if (!global_device_id.ok()) return ToAbslStatus(global_device_id.status());

  StatusOr<DeviceAssignment::LogicalID> logical_id =
      params.device_assn->LogicalIdForDevice(global_device_id.value());
  if (!logical_id.ok()) return ToAbslStatus(logical_id.status());

  se::DeviceMemoryBase result_data = GetDeviceAddress(result);
  params.stream->ThenMemset32(&result_data, logical_id.value().computation_id,
                              /*size=*/4);

  return absl::OkStatus();
}

static bool PartitionId(runtime::ExecutionContext* ctx, void** args,
                        void** attrs, void** rets) {
  static auto* handler = CustomCall::Bind("xla.gpu.partition_id")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .Arg<runtime::FlatMemrefView>()  // result
                             .To<RuntimeChecks()>(PartitionId::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

// Populate mapping from XLA (SE) enums/structs type id to symbol names.
void PopulateXlaGpuTypeIdNames(TypeIDNameRegistry& registry) {
#if GOOGLE_CUDA
  registry.Register<Tagged<se::cuda::BlasLt::Epilogue>>(
      "__type_id_se_cublas_lt_epilogue");
#endif  // GOOGLE_CUDA

  registry.Register<Tagged<se::dnn::ActivationMode>>(
      "__type_id_se_dnn_activation");
  registry.Register<Tagged<se::fft::Type>>("__type_id_se_fft_type");

  registry.Register<Tagged<DotDimensionNumbers>>(
      "__type_id_dot_dimension_numbers");
  registry.Register<Tagged<ConvDimensionNumbers>>(
      "__type_id_conv_dimension_numbers");
  registry.Register<Tagged<ConvBackendConfig>>("__type_id_conv_backend_config");

  RegisterTracingTypeIdNames(registry);
}

void PopulateXlaGpuCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  RegisterKernelLaunchCustomCalls(registry);
  RegisterTracingCustomCalls(registry);

#if GOOGLE_CUDA
  // Graph launch kernels depend on Cuda Graph API.
  RegisterGraphLaunchCustomCalls(registry);
#endif  // GOOGLE_CUDA

  RegisterFftCustomCalls(registry);
  registry.Register("xla.gpu.cholesky", &xla::gpu::Cholesky);
  registry.Register("xla.gpu.collective_permute", &xla::gpu::CollectivePermute);
  registry.Register("xla.gpu.gemm", &xla::gpu::Gemm);

#if GOOGLE_CUDA
  RegisterMatmulCustomCalls(registry);
#endif  // GOOGLE_CUDA

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  registry.Register("xla.gpu.custom_call", &xla::gpu::CustomCall);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

  RegisterConvCustomCalls(registry);

  registry.Register("xla.gpu.memcpy.d2d",
                    &MemcpyFn<MemcpyDirection::kDeviceToDevice>);
  registry.Register("xla.gpu.memcpy.h2d",
                    &MemcpyFn<MemcpyDirection::kHostToDevice>);
  registry.Register("xla.gpu.memcpy.d2h",
                    &MemcpyFn<MemcpyDirection::kDeviceToHost>);
  registry.Register("xla.gpu.memset", &MemsetFn);
  RegisterIoFeedCustomCalls(registry);

  // Collective operations.
  registry.Register("xla.gpu.all_gather", &xla::gpu::AllGather);
  registry.Register("xla.gpu.all_reduce", &xla::gpu::AllReduce);
  registry.Register("xla.gpu.all_reduce_done", &xla::gpu::AllReduceDone);
  registry.Register("xla.gpu.all_reduce_start", &xla::gpu::AllReduceStart);
  registry.Register("xla.gpu.all_to_all", &xla::gpu::AllToAll);
  registry.Register("xla.gpu.reduce_scatter", &xla::gpu::ReduceScatter);
  registry.Register("xla.gpu.partition_id", &xla::gpu::PartitionId);
  registry.Register("xla.gpu.replica_id", &xla::gpu::ReplicaId);
}

}  // namespace gpu
}  // namespace xla
