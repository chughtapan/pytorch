#pragma once

#include <torch/csrc/jit/graph_executor_impl.h>

#include <ATen/core/interned_strings.h>

namespace torch {
namespace jit {

using GraphExecutorImplCreator =
    std::function<GraphExecutorImplBase *(std::shared_ptr<Graph>)>;

TORCH_API extern const Symbol kDefaultExecutor;
struct TORCH_API RegisterGraphExecutorImpl {
  RegisterGraphExecutorImpl(Symbol name, GraphExecutorImplCreator creator);
};

TORCH_API GraphExecutorImplCreator getGraphExecutorImpl();

TORCH_API void setGraphExecutorName(Symbol name);
} // namespace jit
} // namespace torch
