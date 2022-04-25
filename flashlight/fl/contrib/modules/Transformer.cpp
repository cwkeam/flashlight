/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/Transformer.h"
#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace {
fl::Variable transformerInitLinear(int32_t inDim, int32_t outDim) {
  float std = std::sqrt(1.0 / float(inDim));
  return fl::uniform(outDim, inDim, -std, std, af::dtype::f32, true);
}
} // namespace

namespace fl {

Transformer::Transformer(
    int32_t modelDim,
    int32_t headDim,
    int32_t mlpDim,
    int32_t nHeads,
    int32_t bptt,
    float pDropout,
    float pLayerdrop,
    bool useMask,
    bool preLN)
    : nHeads_(nHeads),
      bptt_(bptt),
      pDropout_(pDropout),
      pLayerdrop_(pLayerdrop),
      useMask_(useMask),
      preLN_(preLN),
      w1_(std::make_shared<Linear>(transformerInitLinear(modelDim, mlpDim))),
      w2_(std::make_shared<Linear>(transformerInitLinear(mlpDim, modelDim))),
      wq_(std::make_shared<Linear>(
          transformerInitLinear(modelDim, headDim * nHeads))),
      wk_(std::make_shared<Linear>(
          transformerInitLinear(modelDim, headDim * nHeads))),
      wv_(std::make_shared<Linear>(
          transformerInitLinear(modelDim, headDim * nHeads))),
      wf_(std::make_shared<Linear>(
          transformerInitLinear(headDim * nHeads, modelDim))),
      norm1_(std::make_shared<LayerNorm>(std::vector<int>({0, 3}))),
      norm2_(std::make_shared<LayerNorm>(std::vector<int>({0, 3}))) {
  if (bptt > 0) {
    params_.push_back(
        uniform(2 * bptt - 1, headDim, -0.1, 0.1, af::dtype::f32, true));
  }

  add(w1_);
  add(w2_);
  add(wq_);
  add(wk_);
  add(wv_);
  add(wf_);
  add(norm1_);
  add(norm2_);
  
}

Variable Transformer::mlp(const Variable& input) {
  float pDropout = train_ ? pDropout_ : 0.0;
  return (*w2_)(dropout(relu((*w1_)(input)), pDropout));
}

Variable Transformer::getMask(int32_t n, bool cache) {
  auto mask = af::lower(af::constant(1.0, n, n), true);
  if (cache) {
    auto maskCache = af::upper(af::constant(1.0, n, n));
    mask = af::join(1, maskCache, mask);
  }
  return Variable(af::log(mask), false);
}

Variable Transformer::selfAttention(const std::vector<Variable>& input) {
  std::string savePath = "OUTPUT_TRF.arr";
  const char* savePathChar = savePath.c_str();

  
  // previous step[optionally], input, padMask
  auto encoderInput = input.at(input.size() - 2);

  auto arr = encoderInput.array();
  af::saveArray("selfAttention_input", arr, savePathChar, true); 

  // in case of previous state input[0] has size CxT_prevxB
  int n = input[0].dims(1), bsz = input[0].dims(2);
  double pDrop = train_ ? pDropout_ : 0.0;

  auto q = transpose((*wq_)(encoderInput));
  std::vector<fl::Variable> inputWithState(input.begin(), input.end() - 1);
  auto k = transpose((*wk_)(concatenate(inputWithState, 1)));
  auto v = transpose((*wv_)(concatenate(inputWithState, 1)));

  arr = q.array();
  af::saveArray("selfAttention_q", arr, savePathChar, true); 
  arr = k.array();
  af::saveArray("selfAttention_k", arr, savePathChar, true); 
  arr = v.array();
  af::saveArray("selfAttention_v", arr, savePathChar, true); 

  Variable mask, posEmb;
  if (bptt_ > 0) {
    posEmb =
        tile(params_[0].as(encoderInput.type()), af::dim4(1, 1, nHeads_ * bsz));

    arr = posEmb.array();
    af::saveArray("selfAttention_posEmb", arr, savePathChar, true); 
  }
  if (useMask_ && encoderInput.dims(1) > 1) {
    // mask future if we use the previous state (then n is previous time)
    mask = getMask(n, input.size() == 3);

    arr = mask.array();
    af::saveArray("selfAttention_mask", arr, savePathChar, true); 
  
  }

  int offset = (input.size() == 2) ? 0 : n;

  // time x batch
  fl::Variable padMask;
  if (!input.back().isempty()) {
    auto padMaskArr = input.back().array();
    padMaskArr =
        af::resize(padMaskArr, encoderInput.dims(1), encoderInput.dims(2));
    padMask = fl::Variable(af::log(padMaskArr), false);

    arr = padMask.array();
    af::saveArray("selfAttention_padMask", arr, savePathChar, true); 
  }
  auto result = multiheadAttention(
      q, k, v, posEmb, mask, padMask, nHeads_, pDrop, offset);
    
  arr = result.array();
  af::saveArray("selfAttention_result", arr, savePathChar, true); 

  result = (*wf_)(transpose(result));

  arr = wf_->param(0).array();
  af::saveArray("selfAttention_wf_", arr, savePathChar, true); 

  arr = result.array();
  af::saveArray("selfAttention_result_2", arr, savePathChar, true); 

  return result;
}

std::vector<Variable> Transformer::forward(const std::vector<Variable>& input) {
  std::string savePath = "OUTPUT_TRF.arr";
  const char* savePathChar = savePath.c_str();

  // previous step[optionally], input, padMask
  // padMask should be empty if previous step is provided
  // padMask is expected to have "1" on the used positions and "0" on padded
  // positions
  if (input.size() < 2) {
    throw std::invalid_argument(
        "Invalid inputs for transformer block: there should be at least input and mask");
  }
  auto x = input.at(input.size() - 2);
  if (!input.back().isempty() && x.dims(2) != input.back().dims(1)) {
    throw std::invalid_argument(
        "Invalid inputs for transformer block: input and Mask batch sizes are different");
  }

  float f = 1.0;
  if (train_ && (af::randu(1).scalar<float>() < pLayerdrop_)) {
    f = 0.0;
  }

  auto input_arr = x.array();
  af::saveArray("layer_x", input_arr, savePathChar, true); 



  if (preLN_) {
    auto h = (f * (*norm1_)(selfAttention(input))).as(x.type()) + x;

    input_arr = h.array();
    af::saveArray("layer_h_preln", input_arr, savePathChar, true); 

    return {f * (*norm2_)(mlp(h)).as(h.type()) + h};
  } else {
    // f is used for dropout; it's not some parameter...
    auto selfAttnResult = selfAttention(input);

    input_arr = selfAttnResult.array();
    af::saveArray("layer_selfAttnResult", input_arr, savePathChar, true);

    input_arr = norm1_->param(0).array();
    af::saveArray("layer_norm1_w", input_arr, savePathChar, true); 
    input_arr = norm1_->param(1).array();
    af::saveArray("layer_norm1_b", input_arr, savePathChar, true); 

    auto h = (*norm1_)((f * selfAttnResult.as(x.type())) + x);

    
    input_arr = h.array();
    af::saveArray("layer_h", input_arr, savePathChar, true); 

    

    h = (*norm2_)((f * mlp(h)).as(h.type()) + h);

    input_arr = h.array();
    af::saveArray("layer_h_2", input_arr, savePathChar, true); 

    return {h};
  }
}

std::string Transformer::prettyString() const {
  std::ostringstream ss;
  ss << "Transformer (nHeads: " << nHeads_ << "), "
     << "(pDropout: " << pDropout_ << "), "
     << "(pLayerdrop: " << pLayerdrop_ << "), "
     << "(bptt: " << bptt_ << "), "
     << "(useMask: " << useMask_ << "), "
     << "(preLayerNorm: " << preLN_ << ")";
  return ss.str();
}

std::string Transformer::printWeights() const {
  std::ostringstream ss;
  ss << "Transformer WEIGHTS";
  return ss.str();
}


Transformer::Transformer() {}

} // namespace fl
