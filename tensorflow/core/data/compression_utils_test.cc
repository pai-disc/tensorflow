/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/compression_utils.h"

#include <string>
#include <vector>

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::testing::StatusIs;
using ::testing::HasSubstr;

TEST(CompressionUtilsTest, Exceeds4GB) {
  std::vector<Tensor> element = {
      CreateTensor<int64_t>(TensorShape{1024, 1024, 513})};  // Just over 4GB.
  CompressedElement compressed;
  EXPECT_THAT(CompressElement(element, &compressed),
              StatusIs(error::OUT_OF_RANGE,
                       HasSubstr("exceeding the 4GB Snappy limit")));
}

std::vector<std::vector<Tensor>> TestCases() {
  return {
      // Single int64.
      CreateTensors<int64_t>(TensorShape{1}, {{1}}),
      // Multiple int64s.
      CreateTensors<int64_t>(TensorShape{1}, {{1}, {2}}),
      // Single tstring.
      CreateTensors<tstring>(TensorShape{1}, {{"a"}, {"b"}}),
      // Mix of tstring and int64.
      {CreateTensor<tstring>(TensorShape{1}, {"a"}),
       CreateTensor<int64_t>(TensorShape{1}, {1})},
      // Empty.
      {},
      // Larger int64.
      {CreateTensor<int64_t>(TensorShape{128, 128}),
       CreateTensor<int64_t>(TensorShape{64, 2})},
      // Variants.
      {
          DatasetOpsTestBase::CreateTestVariantTensor(
              {CreateTensor<int64_t>(TensorShape{3, 1}, {1, 2, 3}),
               CreateTensor<tstring>(TensorShape{}, {"abc"})}),
          DatasetOpsTestBase::CreateTestVariantTensor(
              {CreateTensor<int64_t>(TensorShape{3, 1}, {10, 11, 12}),
               CreateTensor<tstring>(TensorShape{}, {"xyz"})}),
      },
  };
}

class ParameterizedCompressionUtilsTest
    : public DatasetOpsTestBase,
      public ::testing::WithParamInterface<std::vector<Tensor>> {};

TEST_P(ParameterizedCompressionUtilsTest, RoundTrip) {
  std::vector<Tensor> element = GetParam();
  CompressedElement compressed;
  TF_ASSERT_OK(CompressElement(element, &compressed));
  std::vector<Tensor> round_trip_element;
  TF_ASSERT_OK(UncompressElement(compressed, &round_trip_element));
  TF_EXPECT_OK(
      ExpectEqual(element, round_trip_element, /*compare_order=*/true));
}

INSTANTIATE_TEST_SUITE_P(Instantiation, ParameterizedCompressionUtilsTest,
                         ::testing::ValuesIn(TestCases()));

class ParameterizedCompressionAndSerializationTest
    : public DatasetOpsTestBase,
      public ::testing::WithParamInterface<std::vector<Tensor>> {};

TEST_P(ParameterizedCompressionAndSerializationTest, RoundTrip) {
  std::vector<Tensor> element = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized_tensors,
                          CompressAndSerialize(element));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Tensor> round_trip_element,
                          DeserializeAndUncompress(serialized_tensors));
  TF_EXPECT_OK(
      ExpectEqual(element, round_trip_element, /*compare_order=*/true));
}

INSTANTIATE_TEST_SUITE_P(Instantiation,
                         ParameterizedCompressionAndSerializationTest,
                         ::testing::ValuesIn(TestCases()));

}  // namespace
}  // namespace data
}  // namespace tensorflow
