/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/lib/audio/feature/Mfsc.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstddef>
#include <numeric>

#include "flashlight/lib/audio/feature/SpeechUtils.h"


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
  std::fstream file;
  file.open("MFSC_OUT.txt", std::ios_base::out);

  for(int i=0;i<input.size();i++)
  {
      file<<input[i]<<std::endl;
  }

  file << std::endl;

  std::cout << "TEST MFSC RUN";
  auto frames = frameSignal(input, this->featParams_);
  if (frames.empty()) {
    return {};
  }

  for(int i=0;i<frames.size();i++)
  {
      file<<frames[i]<<std::endl;
  }

  file << std::endl;

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

  for(int i=0;i<mfscFeat.size();i++)
  {
      file<<mfscFeat[i]<<std::endl;
  }

  file << std::endl;


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

  file.close();

  return derivatives_.apply(mfscFeat, numFeat);
}

std::vector<float> Mfsc::mfscImpl(std::vector<float>& frames) {
  std::fstream file;
  file.open("MFSCImpl_OUT.txt", std::ios_base::out);

  for(int i=0;i<frames.size();i++)
  {
      file<<frames[i]<<std::endl;
  }
  file << std::endl;


  std::cout << "MFSC IMPL RUN";
  auto powspectrum = this->powSpectrumImpl(frames);

  for(int i=0;i<powspectrum.size();i++)
  {
      file<<powspectrum[i]<<std::endl;
  }
  file << std::endl;


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

  for(int i=0;i<triflt.size();i++)
  {
      file<<triflt[i]<<std::endl;
  }
  file << std::endl;


  std::transform(triflt.begin(), triflt.end(), triflt.begin(), [](float x) {
    return std::log(x);
  });

  for(int i=0;i<triflt.size();i++)
  {
      file<<triflt[i]<<std::endl;
  }
  file << std::endl;

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
