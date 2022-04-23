/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/lib/audio/feature/Mfsc.h"
#include <iostream>
#include <algorithm>
#include <cstddef>
#include <numeric>
#include "flashlight/fl/common/Utils.h"
#include "flashlight/lib/audio/feature/SpeechUtils.h"
using namespace af;

namespace fl {
namespace lib {
namespace audio {

Mfsc::Mfsc(const FeatureParams& params)
    : PowerSpectrum(params),
      triFltBank_(
          params.numFilterbankChans,
          params.filterFreqResponseLen(),
          params.samplingFreq,
          params.lowFreqFilterbank,
          params.highFreqFilterbank,
          FrequencyScale::MEL),
      derivatives_(params.deltaWindow, params.accWindow) {
  validateMfscParams();
}

std::vector<float> Mfsc::apply(const std::vector<float>& input) {
  std::string savePath = "OUTPUT_MFSC.arr";
  const char* savePathChar = savePath.c_str();

  auto input_arr = af::array(input.size(), input.data());
  af::saveArray("msfc_1_input", input_arr, savePathChar, true);

  std::cout << "TEST MFSC RUN";
  auto frames = frameSignal(input, this->featParams_);
  if (frames.empty()) {
    return {};
  }

  input_arr = af::array(frames.size(), frames.data());
  af::saveArray("msfc_2_frames", input_arr, savePathChar, true);


  int nSamples = this->featParams_.numFrameSizeSamples();
  int nFrames = frames.size() / nSamples;

  std::vector<float> energy(nFrames);

  // this if statement is faise.
  if (this->featParams_.useEnergy && this->featParams_.rawEnergy) {
    std::cout << "USE ENERGY AND RAW ENERGY";
    for (size_t f = 0; f < nFrames; ++f) {
      auto begin = frames.data() + f * nSamples;
      energy[f] = std::log(std::max(
          std::inner_product(
              begin, begin + nSamples, begin, static_cast<float>(0.0)),
          std::numeric_limits<float>::lowest()));
    }
  }

  auto mfscFeat = mfscImpl(frames);

  input_arr = af::array(mfscFeat.size(), mfscFeat.data());
  af::saveArray("msfc_3_mfscFeat", input_arr, savePathChar, true);

  auto numFeat = this->featParams_.numFilterbankChans;

  // this if statement is false
  if (this->featParams_.useEnergy) {
    std::cout << "USE ENERGY2";
    if (!this->featParams_.rawEnergy) {
      std::cout << "NO RAW ENERGY";
      for (size_t f = 0; f < nFrames; ++f) {
        auto begin = frames.data() + f * nSamples;
        energy[f] = std::log(std::max(
            std::inner_product(
                begin, begin + nSamples, begin, static_cast<float>(0.0)),
            std::numeric_limits<float>::lowest()));
      }
    }
    std::vector<float> newMfscFeat(mfscFeat.size() + nFrames);
    for (size_t f = 0; f < nFrames; ++f) {
      size_t start = f * numFeat;
      newMfscFeat[start + f] = energy[f];
      std::copy(
          mfscFeat.data() + start,
          mfscFeat.data() + start + numFeat,
          newMfscFeat.data() + start + f + 1);
    }
    std::swap(mfscFeat, newMfscFeat);
    ++numFeat;
  }
  // Derivatives will not be computed if windowsize < 0
  return derivatives_.apply(mfscFeat, numFeat);
}

std::vector<float> Mfsc::mfscImpl(std::vector<float>& frames) {
  std::string savePath = "OUTPUT_MFSC.arr";
  const char* savePathChar = savePath.c_str();

  auto arr = af::array(frames.size(), frames.data());
  af::saveArray("2_1_frames_msfcImpl", arr, savePathChar, true);



  std::cout << "MFSC IMPL RUN";
  auto powspectrum = this->powSpectrumImpl(frames);

  arr = af::array(powspectrum.size(), powspectrum.data());
  af::saveArray("2_2_powspectrum_msfcImpl", arr, savePathChar, true);

  // this doesn't run
  if (this->featParams_.usePower) {
    std::cout << "USE POWER";
    std::transform(
        powspectrum.begin(),
        powspectrum.end(),
        powspectrum.begin(),
        [](float x) { return x * x; });
  }

  auto triflt = triFltBank_.apply(powspectrum, this->featParams_.melFloor);

  std::cout << "melFLOOR=" << this->featParams_.melFloor;

  arr = af::array(triflt.size(), triflt.data());
  af::saveArray("2_3_triflt_msfcImpl", arr, savePathChar, true);


  std::transform(triflt.begin(), triflt.end(), triflt.begin(), [](float x) {
    return std::log(x);
  });

  arr = af::array(triflt.size(), triflt.data());
  af::saveArray("2_4_log_msfcImpl", arr, savePathChar, true);


  return triflt;
}

int Mfsc::outputSize(int inputSz) {
  return this->featParams_.mfscFeatSz() * this->featParams_.numFrames(inputSz);
}

void Mfsc::validateMfscParams() const {
  this->validatePowSpecParams();
  if (this->featParams_.numFilterbankChans <= 0) {
    throw std::invalid_argument("Mfsc: numFilterbankChans must be positive");
  } else if (this->featParams_.melFloor <= 0.0) {
    throw std::invalid_argument("Mfsc: melfloor must be positive");
  }
}
} // namespace audio
} // namespace lib
} // namespace fl
