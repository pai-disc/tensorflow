/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/auto_shard.h"

#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/graph_test_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

using ::tensorflow::test::function::GDef;
using ::tensorflow::test::function::NDef;
using ::testing::UnorderedElementsAre;

TEST(RewriteBatchTest, InfiniteSource) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("repeat_count", "Const", {}, {{"value", -1}, {"dtype", DT_INT32}}),
      NDef("repeat", "RepeatDataset", {"tf_record", "repeat_count"},
           {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      NDef("batch", "BatchDatasetV2",
           {"repeat", "batch_size", "drop_remainder"},
           {{"_cardinality", data::kInfiniteCardinality},
            {"parallel_copy", false},
            {"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("num_replicas", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("rebatch", "RebatchDataset", {"batch", "num_replicas"},
           {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("Sink", "Identity", {"rebatch"}, {}),
  });

  item.fetch.push_back("Sink");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_TRUE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                   &ineligible_reason))
      << absl::StrJoin(ineligible_reason, ",");
}

TEST(RewriteBatchTest, FiniteSourceNoDropRemainder) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {},
           {{"value", false}, {"dtype", DT_BOOL}}),
      NDef("batch", "BatchDatasetV2",
           {"repeat", "batch_size", "drop_remainder"},
           {{"_cardinality", data::kUnknownCardinality},
            {"parallel_copy", false},
            {"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("num_replicas", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("rebatch", "RebatchDataset", {"batch", "num_replicas"},
           {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("Sink", "Identity", {"rebatch"}, {}),
  });

  item.fetch.push_back("Sink");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_TRUE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                   &ineligible_reason))
      << absl::StrJoin(ineligible_reason, ",");
}

TEST(RewriteBatchTest, FiniteSourceDropRemainder) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      NDef("batch", "BatchDatasetV2",
           {"repeat", "batch_size", "drop_remainder"},
           {{"_cardinality", 1337},
            {"parallel_copy", false},
            {"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("num_replicas", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("rebatch", "RebatchDataset", {"batch", "num_replicas"},
           {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("Sink", "Identity", {"rebatch"}, {}),
  });

  item.fetch.push_back("Sink");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_FALSE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                    &ineligible_reason));
  EXPECT_THAT(ineligible_reason,
              UnorderedElementsAre("BATCH_DROP_REMAINDER_NOT_INFINITE"));
}

TEST(RewriteBatchTest, UnknownCardinalitySourceDropRemainder) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      NDef("batch", "BatchDatasetV2",
           {"repeat", "batch_size", "drop_remainder"},
           {{"_cardinality", data::kUnknownCardinality},
            {"parallel_copy", false},
            {"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("num_replicas", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("rebatch", "RebatchDataset", {"batch", "num_replicas"},
           {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("Sink", "Identity", {"rebatch"}, {}),
  });

  item.fetch.push_back("Sink");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_FALSE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                    &ineligible_reason));
  EXPECT_THAT(ineligible_reason,
              UnorderedElementsAre("BATCH_DROP_REMAINDER_NOT_INFINITE"));
}

TEST(RewriteBatchTest, FiniteSourceDropRemainderUnknown) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "RandomBool", {}, {}),
      NDef("batch", "BatchDatasetV2",
           {"repeat", "batch_size", "drop_remainder"},
           {{"_cardinality", 1337},
            {"parallel_copy", false},
            {"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("num_replicas", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("rebatch", "RebatchDataset", {"batch", "num_replicas"},
           {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("Sink", "Identity", {"rebatch"}, {}),
  });

  item.fetch.push_back("Sink");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  absl::flat_hash_map<std::string, int64_t> cardinalities{
      {"tf_record", data::kUnknownCardinality},
      {"batch", data::kUnknownCardinality},
      {"rebatch", data::kUnknownCardinality},
  };
  std::vector<std::string> ineligible_reason;
  EXPECT_FALSE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                    &ineligible_reason));
  EXPECT_THAT(ineligible_reason,
              UnorderedElementsAre("BATCH_DROP_REMAINDER_UNKNOWN"));
}

TEST(RewriteBatchTest, DropRemainderCardinalityNotAvailable) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {}, {{"value", true}}),
      NDef("batch", "BatchDatasetV2",
           {"repeat", "batch_size", "drop_remainder"},
           {{"parallel_copy", false},
            {"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("num_replicas", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("rebatch", "RebatchDataset", {"batch", "num_replicas"},
           {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("Sink", "Identity", {"rebatch"}, {}),
  });

  item.fetch.push_back("Sink");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_FALSE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                    &ineligible_reason));
  EXPECT_THAT(ineligible_reason,
              UnorderedElementsAre("BATCH_CARDINALITY_NOT_AVAILABLE"));
}

TEST(RewriteBatchTest, OpNotSupported) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      NDef("batch", "BatchDatasetV2",
           {"repeat", "batch_size", "drop_remainder"},
           {{"_cardinality", 1337},
            {"_drop_remainder", true},
            {"parallel_copy", false},
            {"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("take_count", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      graph_tests_utils::MakeTakeNode("take", "batch", "take_count"),
      NDef("num_replicas", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("rebatch", "RebatchDataset", {"take", "num_replicas"},
           {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("Sink", "Identity", {"rebatch"}, {}),
  });

  item.fetch.push_back("Sink");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_FALSE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                    &ineligible_reason));
  EXPECT_THAT(ineligible_reason,
              UnorderedElementsAre("OP_NOT_SUPPORTED_TakeDataset",
                                   "BATCH_DROP_REMAINDER_NOT_INFINITE"));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
