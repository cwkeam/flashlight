/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/data/FeatureTransforms.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "flashlight/app/asr/data/Utils.h"
#include "flashlight/lib/audio/feature/Mfcc.h"
#include "flashlight/lib/audio/feature/Mfsc.h"
#include "flashlight/lib/audio/feature/PowerSpectrum.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

using namespace fl::lib;
using namespace fl::lib::audio;
using fl::lib::text::Dictionary;
using fl::lib::text::LexiconMap;
using fl::lib::text::packReplabels;

namespace {

size_t getSfxSeed() {
  auto a = fl::lib::getProcessId();
  auto b = fl::lib::getThreadId();
  // Create an unique hash from thread, process id
  // using Cantor pairing function
  return 0.5 * (a + b) * (a + b + 1) + b;
}

class StartSfxCounter {
 public:
  explicit StartSfxCounter(int n) : iters_(n) {}
  bool decrementAndCheck() {
    std::lock_guard<std::mutex> lock(mutex_);
    iters_ = iters_ > 0 ? iters_ - 1 : iters_;
    return iters_ <= 0;
  }
 private:
  int iters_;
  std::mutex mutex_;
};

} // namespace

namespace fl {
namespace app {
namespace asr {

fl::Dataset::DataTransformFunction inputFeatures(
    const FeatureParams& params,
    const FeatureType& featureType,
    const std::pair<int, int>& localNormCtx,
    const std::vector<sfx::SoundEffectConfig>& sfxConf /* = {} */,
    const int sfxStartUpdate /* = 0 */) {
  auto sfxCounter = std::make_shared<StartSfxCounter>(sfxStartUpdate);
  return [params, featureType, localNormCtx, sfxConf, sfxCounter](
             void* data, af::dim4 dims, af::dtype type) {
    if (type != af::dtype::f32) {
      throw std::invalid_argument("Invalid input type");
    }
    if (dims[2] * dims[3] != 1) {
      throw std::invalid_argument(
          "'inputFeatures': Invalid input dims . Expected 2d array - Channels x T");
    }
    auto channels = dims[0];
    std::vector<float> input(dims.elements());

    const char* savePathChar = "/content/features.arr";

    auto input_arr = af::array(input.size(), input.data());
    af::saveArray("input_raw", input_arr, savePathChar, true); 


    std::copy_n(static_cast<const float*>(data), input.size(), input.data());
    if (channels > 1) {
      input = transpose2d(input, dims[1], channels);

      auto input_transpose = af::array(input.size(), input.data());
      af::saveArray("input_transpose2d", input_transpose, savePathChar, true); 

    }


    input_arr = af::array(input.size(), input.data());
    af::saveArray("input_raw_2", input_arr, savePathChar, true); 


    if (!sfxConf.empty() && sfxCounter->decrementAndCheck()) {
      if (channels > 1) {
        throw std::invalid_argument(
            "'inputFeatures': Invalid input dims. sound effect supports a single channel audio");
      }
      thread_local auto seed = getSfxSeed();
      thread_local std::shared_ptr<sfx::SoundEffect> sfx =
          sfx::createSoundEffect(sfxConf, seed);
      sfx->apply(input);
    }

    input_arr = af::array(input.size(), input.data());
    af::saveArray("input_raw_3", input_arr, savePathChar, true); 

    std::vector<float> output;
    int featSz = 1;
    if (featureType == FeatureType::POW_SPECTRUM) {
      thread_local PowerSpectrum powspec(params);
      featSz = params.powSpecFeatSz();
      output = powspec.batchApply(input, channels);

      auto output_arr = af::array(output.size(), output.data());
      af::saveArray("output_POW_SPECTRUM", output_arr, savePathChar, true); 

    } else if (featureType == FeatureType::MFSC) {
      thread_local Mfsc mfsc(params);
      featSz = params.mfscFeatSz();
      output = mfsc.batchApply(input, channels);

      auto output_arr = af::array(output.size(), output.data());
      af::saveArray("output_MFSC", output_arr, savePathChar, true); 

    } else if (featureType == FeatureType::MFCC) {
      thread_local Mfcc mfcc(params);
      featSz = params.mfccFeatSz();
      output = mfcc.batchApply(input, channels);

      auto output_arr = af::array(output.size(), output.data());
      af::saveArray("output_MFCC", output_arr, savePathChar, true); 

    } else {
      // use raw audio
      output = input; // T X CHANNELS (Col Major)
      auto output_arr = af::array(output.size(), output.data());
      af::saveArray("output_RAW", output_arr, savePathChar, true);
    }

    auto T = output.size() / (featSz * channels);
    // Before: FEAT X FRAMES X CHANNELS  (Col Major)
    output = transpose2d(output, T, featSz, channels);

    auto output_arr = af::array(output.size(), output.data());
    af::saveArray("output_T_then_transpose", output_arr, savePathChar, true);
    // After: FRAMES X FEAT X CHANNELS  (Col Major)
    
    if (localNormCtx.first > 0 || localNormCtx.second > 0) {
      output =
          localNormalize(output, localNormCtx.first, localNormCtx.second, T);
      
      auto output_arr = af::array(output.size(), output.data());
      af::saveArray("output_localNormalize", output_arr, savePathChar, true);
    } else {
      output = normalize(output);
      auto output_arr = af::array(output.size(), output.data());
      af::saveArray("output_normalize", output_arr, savePathChar, true);
    }
    
    auto finalOutput = af::array(T, featSz, channels, output.data());
    af::saveArray("output_finalOutput", finalOutput, savePathChar, true);

    return finalOutput;
  };
}

// target
fl::Dataset::DataTransformFunction targetFeatures(
    const Dictionary& tokenDict,
    const LexiconMap& lexicon,
    const TargetGenerationConfig& config) {
  return [tokenDict, lexicon, config](
             void* data, af::dim4 dims, af::dtype /* unused */) {
    std::string transcript(
        static_cast<char*>(data), static_cast<char*>(data) + dims.elements());
    auto words = splitOnWhitespace(transcript, true);
    auto target = wrd2Target(
        words,
        lexicon,
        tokenDict,
        config.wordSeparator_,
        config.targetSamplePct_,
        config.fallbackToLetterWordSepLeft_,
        config.fallbackToLetterWordSepRight_,
        config.skipUnk_);
    auto tgtVec = tokenDict.mapEntriesToIndices(target);
    if (!config.surround_.empty()) {
      // add surround token at the beginning and end of target
      // only if begin/end tokens are not surround
      auto idx = tokenDict.getIndex(config.surround_);
      if (tgtVec.empty() || tgtVec.back() != idx) {
        tgtVec.emplace_back(idx);
      }
      if (tgtVec.size() > 1 && tgtVec.front() != idx) {
        tgtVec.emplace_back(idx);
        std::rotate(tgtVec.begin(), tgtVec.end() - 1, tgtVec.end());
      }
    }
    if (config.replabel_ > 0) {
      tgtVec = packReplabels(tgtVec, tokenDict, config.replabel_);
    }
    if (config.criterion_ == kAsgCriterion) {
      dedup(tgtVec);
    }
    if (config.eosToken_) {
      tgtVec.emplace_back(tokenDict.getIndex(kEosToken));
    }
    if (tgtVec.empty()) {
      // support empty target
      return af::array().as(s32);
    }
    return af::array(tgtVec.size(), tgtVec.data());
  };
}

fl::Dataset::DataTransformFunction wordFeatures(const Dictionary& wrdDict) {
  return [wrdDict](void* data, af::dim4 dims, af::dtype /* unused */) {
    std::string transcript(
        static_cast<char*>(data), static_cast<char*>(data) + dims.elements());
    auto words = splitOnWhitespace(transcript, true);
    auto wrdVec = wrdDict.mapEntriesToIndices(words);
    if (wrdVec.empty()) {
      // support empty target
      return af::array().as(s32);
    }
    return af::array(wrdVec.size(), wrdVec.data());
  };
}
} // namespace asr
} // namespace app
} // namespace fl
