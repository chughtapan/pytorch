#include <torch/csrc/jit/passes/onnx/constant_fold.h>
#include <c10/util/Exception.h>

#include <c10/util/Optional.h>
#include <algorithm>

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

namespace {

using ParamMap = std::map<std::string, at::Tensor>;
using ValueToParamPairMap =
    std::map<Value*, std::pair<std::string, at::Tensor>>;

std::unordered_map<int, at::ScalarType> onnxTypeToScalarTypeMap = {
    // Only conversion of ONNX numeric types is included here.
    // Unsigned ONNX types are mapped to the next higher signed
    // ScalarType type.
    {1, at::kFloat},
    {2, at::kByte},
    {3, at::kChar},
    {4, at::kInt},
    {5, at::kShort},
    {6, at::kInt},
    {7, at::kLong},
    {10, at::kFloat},
    {11, at::kDouble},
    {12, at::kLong},
};

ValueToParamPairMap buildValueToParamsMap(
    Block* b,
    const ParamMap& paramsDict) {
  ValueToParamPairMap valsToParamsMap;
  for (auto& input : b->inputs()) {
    auto it = paramsDict.find(input->debugName());
    if (it != paramsDict.end()) {
      valsToParamsMap.emplace(input, *it);
    }
  }
  return valsToParamsMap;
}

void buildParamsMapFromValueToParamsMap(
    const ValueToParamPairMap& valsToParamsMap,
    ParamMap& paramsDict) {
  paramsDict.clear();
  for (const auto& nameTensorParamPair : valsToParamsMap) {
    paramsDict.insert(nameTensorParamPair.second);
  }
}

void eraseUnusedBlockInputs(Block* b) {
  for (size_t i_1 = b->inputs().size(); i_1 > 0; --i_1) {
    size_t i = i_1 - 1;
    if (!b->inputs().at(i)->hasUses()) {
      b->eraseInput(i);
    }
  }
}

void handleNegativeStartEndIndex(int64_t& start, int64_t& end, int64_t& axis,
                                 c10::IntArrayRef tensorSizes) {
  if (start < 0) {
    start = tensorSizes[axis] + start;
  }
  if (end < 0) {
    end = tensorSizes[axis] + end;
  }
  // index higher than dimension is treated as the end.
  if (end > tensorSizes[axis]) {
    end = tensorSizes[axis];
  }      
}

c10::optional<at::Tensor> runTorchSlice_opset9(const Node* node, 
                     std::vector<at::Tensor>& inputTensorValues) {
  assert(inputTensorValues.size() == 1);
  if (inputTensorValues.size() != 1) {
    std::cerr << "Warning: Constant folding - Invalid number of inputs found for opset 9 onnx::Slice op. "
              << "Constant folding not applied." << std::endl;
    return c10::nullopt;
  }
  if (!(node->hasAttributeS("starts") && node->hasAttributeS("ends"))) {
    return c10::nullopt;
  }
  auto startsAttr = node->is(attr::starts);
  auto endsAttr = node->is(attr::ends);
  if (startsAttr.size() != endsAttr.size()) {
    return c10::nullopt;
  }
  std::vector<int64_t> axesAttr;
  if (node->hasAttributeS("axes")) {
    axesAttr = node->is(attr::axes);
  } else {
    axesAttr.resize(startsAttr.size());
    std::iota(axesAttr.begin(), axesAttr.end(), 0);
  }
  auto updated_val = inputTensorValues[0];
  for (size_t i = 0; i < axesAttr.size(); ++i) {
    // ONNX slice accepts negative starts and ends values.
    int64_t axis = axesAttr[i], start = startsAttr[i], end = endsAttr[i];
    handleNegativeStartEndIndex(start, end, axis, updated_val.sizes());
    int64_t length = end - start;
    if (length < 0 || start > updated_val.sizes()[axis] - length)
      return c10::nullopt;
    updated_val = at::narrow(updated_val, axis, start, length);
  }
  return c10::optional<at::Tensor>(updated_val);
}

c10::optional<at::Tensor> runTorchSlice_opset10(const Node* node, 
                     std::vector<at::Tensor>& inputTensorValues) {
  if (inputTensorValues.size() < 3 || inputTensorValues.size() > 5) {
    std::cerr << "Warning: Constant folding - Invalid number of inputs found for opset 10 onnx::Slice op. "
              << "Constant folding not applied." << std::endl;
    return c10::nullopt;
  }
  // Checking validity of 'starts' and 'ends' input
  if (inputTensorValues[1].sizes().size() != 1 || inputTensorValues[2].sizes().size() != 1) {
    std::cerr << "Warning: Constant folding - Invalid 'starts' or 'ends' inputs found for opset 10 onnx::Slice op. "
              << "Constant folding not applied." << std::endl;
    return c10::nullopt;
  }            
  if (inputTensorValues[1].sizes()[0] != inputTensorValues[2].sizes()[0] ) {
    // Number of elements of 'starts' and 'ends' 1-D input tensors should be the same
    return c10::nullopt;
  }
  // Checking 'axes' input, if available.
  std::vector<int64_t> axes;
  if (inputTensorValues.size() > 3) { 
    if (inputTensorValues[3].sizes().size() != 1) {
      std::cerr << "Warning: Constant folding - Invalid 'axes' input found for opset 10 onnx::Slice op. "
                << "Constant folding not applied." << std::endl;
      return c10::nullopt;
    }
    if (inputTensorValues[3].sizes()[0] != inputTensorValues[1].sizes()[0] ) {
      // Number of elements of 'axes' and 'ends' 1-D input tensors should be the same
      std::cerr << "Warning: Constant folding - Invalid 'axes' or 'ends' inputs found for opset 10 onnx::Slice op. "
                << "Constant folding not applied." << std::endl;
      return c10::nullopt;
    }
    auto axes_a = inputTensorValues[3].accessor<int64_t, 1>();
    axes.reserve(inputTensorValues[3].sizes()[0]); 
    for (size_t i = 0; i < inputTensorValues[3].sizes()[0]; ++i) {
      axes[i] = axes_a[i];
    }
  }
  else {
    axes = std::vector<int64_t>(inputTensorValues[1].sizes()[0], 0);
  }
  // Checking 'steps' input, if available.
  if (inputTensorValues.size() > 4) {
    if (inputTensorValues[4].sizes().size() != 1) {
      std::cerr << "Warning: Constant folding - Invalid 'steps' input found for opset 10 onnx::Slice op. "
                << "Constant folding not applied." << std::endl;
      return c10::nullopt;
    }
    if (inputTensorValues[4].sizes()[0] != inputTensorValues[1].sizes()[0] ) {
      // Number of elements of 'steps' and 'ends' 1-D input tensors should be the same
      std::cerr << "Warning: Constant folding - Invalid 'steps' or 'ends' inputs found for opset 10 onnx::Slice op. "
                << "Constant folding not applied." << std::endl;
      return c10::nullopt;
    }
    auto steps_a = inputTensorValues[4].accessor<int64_t, 1>();
    for (size_t i = 0; i < inputTensorValues[4].sizes()[0]; ++i) {
      // Only steps == 1 are supported for constant-folding.
      if (steps_a[i] != 1) {
        std::cerr << "Warning: Constant folding - Only steps=1 can be constant folded for opset 10 onnx::Slice op. "
                << "Constant folding not applied." << std::endl;
        return c10::nullopt;
      }
    }
  }
  auto starts_a = inputTensorValues[1].accessor<int64_t, 1>();
  auto ends_a = inputTensorValues[2].accessor<int64_t, 1>();
  auto updated_val = inputTensorValues[0];
  for (size_t i = 0; i < inputTensorValues[1].sizes()[0]; ++i) {
    // ONNX slice accepts negative starts and ends values.
    int64_t start = starts_a[i], end = ends_a[i], axis = axes[i];
    handleNegativeStartEndIndex(start, end, axis, updated_val.sizes());
    int64_t length = end - start;
    if (length < 0 || start > updated_val.sizes()[axis] - length)
      return c10::nullopt;
    updated_val = at::narrow(updated_val, axis, start, length);
  }
  return c10::optional<at::Tensor>(updated_val);
}

c10::optional<at::Tensor> runTorchBackendForOnnx(
    const Node* node,
    std::vector<at::Tensor>& inputTensorValues,
    int opset_version) {
  at::Tensor updated_val;
  if (node->kind() == onnx::Slice) {
    if (opset_version == 9) {
      return runTorchSlice_opset9(node, inputTensorValues);
    }
    else if (opset_version == 10) {
      return runTorchSlice_opset10(node, inputTensorValues);
    }
    else {
      std::cerr << "Warning: Constant folding - unsupported opset version. "
              << "Constant folding not applied." << std::endl;
      return c10::nullopt;
    }
  } else if (node->kind() == onnx::Concat) {
    if (!node->hasAttributeS("axis")) {
      return c10::nullopt;
    }
    updated_val =
        at::cat(at::TensorList(inputTensorValues), node->i(attr::axis));
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Unsqueeze) {
    assert(inputTensorValues.size() == 1);
    if (!node->hasAttributeS("axes")) {
      return c10::nullopt;
    }
    updated_val = inputTensorValues[0];
    for (auto axis : node->is(attr::axes)) {
      updated_val = at::unsqueeze(updated_val, axis);
    }
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Transpose) {
    assert(inputTensorValues.size() == 1);
    if (!node->hasAttributeS("perm")) {
      return c10::nullopt;
    }
    updated_val = inputTensorValues[0].permute(node->is(attr::perm));
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Cast) {
    assert(inputTensorValues.size() == 1);
    if (node->hasAttributeS("to") &&
        onnxTypeToScalarTypeMap.find(node->i(attr::to)) !=
            onnxTypeToScalarTypeMap.end()) {
      updated_val =
          inputTensorValues[0].to(onnxTypeToScalarTypeMap[node->i(attr::to)]);
      return c10::optional<at::Tensor>(updated_val);
    }
    return c10::nullopt;
  } else {
    return c10::nullopt;
  }
}

bool isConstant(Value* val, const ValueToParamPairMap& valsToParamsMap) {
  auto parentNode = val->node();
  return (parentNode->kind() == prim::Param &&
          valsToParamsMap.find(val) !=
              valsToParamsMap
                  .end()) || // Checks val is a parameter and not a real input
      (parentNode->kind() == onnx::Constant && !parentNode->mustBeNone() &&
       parentNode->kindOf(attr::value) ==
           AttributeKind::t); // Check other types?
}

std::vector<at::Tensor> getValues(
    Node* node,
    const ValueToParamPairMap& valsToParamsMap) {
  size_t numInputs = node->inputs().size();
  std::vector<at::Tensor> inputTensorValues;
  inputTensorValues.reserve(numInputs);
  for (auto val : node->inputs()) {
    if (val->node()->kind() == prim::Param) {
      auto itr = valsToParamsMap.find(val);
      if (itr == valsToParamsMap.end()) {
        throw std::runtime_error(
            "getValues: Input value not found amongst constant parameters.");
      }
      inputTensorValues.push_back(itr->second.second);
    } else if (val->node()->kind() == onnx::Constant) {
      inputTensorValues.push_back(val->node()->t(attr::value));
    } else {
      throw std::runtime_error(
          "getValues: Unsupported kind of constant node found.");
    }
  }
  AT_ASSERT(inputTensorValues.size() == numInputs);
  return inputTensorValues;
}

void eraseUnusedValuesFromMap(ValueToParamPairMap& valsToParamsMap) {
  auto it = valsToParamsMap.begin();
  while (it != valsToParamsMap.end()) {
    if (!it->first->hasUses()) {
      it = valsToParamsMap.erase(it);
    } else {
      ++it;
    }
  }
}

bool areNodeInputsConstant(
    Node* node,
    const ValueToParamPairMap& valsToParamsMap) {
  return std::all_of(
      node->inputs().begin(),
      node->inputs().end(),
      [&valsToParamsMap](Value* v) { return isConstant(v, valsToParamsMap); });
}

std::vector<Node*> getOnnxConstParentsToRemove(Node* node) {
  std::vector<Node*> parentNodes;
  for (auto val : node->inputs()) {
    // If the parent of 'node' is an onnx::Constant node,
    // and 'node' is the only downstream node it serves (this
    // is important), then push it in the list to remove.
    if (val->node()->kind() == onnx::Constant &&
        val->uses().size() == 1) {
          parentNodes.push_back(val->node());
    }
  }
  return parentNodes;
}

} // Anonymous namespace

// This method updates the block in-place to fold all the one-time
// constant-based computations/ops into an initializer node.
//
// NB: This is not constant folding in the traditional sense, as we
// don't try particularly hard to evaluate operations on constant nodes.
// This is more of a partial evaluation  analysis, where operations on constant
// nodes can be lifted so we run them earlier, before the usual parameters are
// known.
void ConstantFoldONNX(Block* b, ParamMap& paramsDict, int opset_version) {
  if (opset_version != 9 && opset_version != 10) {
    // Number of elements of 'axes' and 'ends' 1-D input tensors should be the same
    std::cerr << "Warning: Constant folding supported for only opsets 9 and 10. "
              << "Constant folding not applied." << std::endl;
    return;
  }
  AT_ASSERT(b->param_node());
  auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
  // Only the root block is constant-folded. Folding nested blocks is
  // not supported for now.
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    auto node = *it;
    if (node->outputs().size() > 1) {
      // Constant folding for multiple-output nodes not supported. Skip it.
      continue;
    }
    if (!areNodeInputsConstant(node, valsToParamsMap)) {
      // If all the inputs to this node are not either parameter or
      // onnx::Constant, then skip this node.
      continue;
    }

    auto inputTensorValues = getValues(node, valsToParamsMap);
    if (inputTensorValues.empty()) {
      // This is a terminal node with no inputs, such as onnx::Constant. Skip
      // it.
      continue;
    }
    auto updatedValWrapped = runTorchBackendForOnnx(node, inputTensorValues, opset_version);
    if (updatedValWrapped == c10::nullopt) {
      // Constant folding is not supported for this op. Skip it.
      continue;
    }
    // Create a new input to the block (prim::Param node output). Add a
    // corresponding entryin valToParamMap. Replace the downstream inputs
    // with this value, and disconnect all the input values of the folded node.
    at::Tensor updatedVal = *updatedValWrapped;
    auto newSourceNodeOutput = b->addInput();
    valsToParamsMap.insert(
        {newSourceNodeOutput,
         std::make_pair(newSourceNodeOutput->debugName(), updatedVal)});
    newSourceNodeOutput->inferTypeFrom(updatedVal);
    node->outputs().at(0)->replaceAllUsesWith(newSourceNodeOutput);

    // Next we remove the current node that has been replaced by
    // an initializer. But before we start de-wiring this node,
    // we check if any parents of this nodes were onnx::Constant
    // and remove them first (following proper sequence as shown
    // below), and then remove the current node. If the parent was
    // an initializer (not onnx::Constant) then they are all removed
    // by eraseUnusedBlockInputs() call (below) outside the loop.
    auto onnxConstParents = getOnnxConstParentsToRemove(node);
    node->removeAllInputs();
    for (auto* n : onnxConstParents) {
      n->destroy();
    }
    it.destroyCurrent();
  }
  eraseUnusedValuesFromMap(valsToParamsMap);
  eraseUnusedBlockInputs(b);
  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);
  return;
}

} // namespace jit
} // namespace torch
