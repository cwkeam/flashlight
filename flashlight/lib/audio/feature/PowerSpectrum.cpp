/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/lib/audio/feature/PowerSpectrum.h"

#include <fftw3.h>

#include <fstream>
#include <iostream>

#include <algorithm>
#include <cstddef>
#include <unordered_map>

#include "flashlight/lib/audio/feature/SpeechUtils.h"

namespace fl {
namespace lib {
namespace audio {

PowerSpectrum::PowerSpectrum(const FeatureParams& params)
    : featParams_(params),
      dither_(params.ditherVal),
      preEmphasis_(params.preemCoef, params.numFrameSizeSamples()),
      windowing_(params.numFrameSizeSamples(), params.windowType) {
  validatePowSpecParams();
  auto nFFt = featParams_.nFft();
  inFftBuf_.resize(nFFt, 0.0);
  outFftBuf_.resize(2 * nFFt);
  fftPlan_ = std::make_unique<fftw_plan>(fftw_plan_dft_r2c_1d(
      nFFt, inFftBuf_.data(), (fftw_complex*)outFftBuf_.data(), FFTW_MEASURE));
}

std::vector<float> PowerSpectrum::apply(const std::vector<float>& input) {
  auto frames = frameSignal(input, featParams_);
  if (frames.empty()) {
    return {};
  }
  return powSpectrumImpl(frames);
}

std::vector<float> PowerSpectrum::powSpectrumImpl(std::vector<float>& frames) {
  std::fstream file;
  file.open("powSpectrum_OUT.txt", std::ios_base::out);

  // 1. pow input
  for(int i=0;i<frames.size();i++)
  {
      file<<frames[i]<<std::endl;
  }
  file << std::endl;


  int nSamples = featParams_.numFrameSizeSamples();
  int nFrames = frames.size() / nSamples;
  int nFft = featParams_.nFft();
  int K = featParams_.filterFreqResponseLen();

  std::cout << "nSamples" << nSamples;
  std::cout << "nFrames" << nFrames;
  std::cout << "nFft" << nFft;
  std::cout << "K" << K;


  if (featParams_.ditherVal != 0.0) {
    std::cout << "DITHER RAN";
    frames = dither_.apply(frames);
  }
  if (featParams_.zeroMeanFrame) {
    std::cout << "ZERO RAN";

    for (size_t f = 0; f < nFrames; ++f) {
      auto begin = frames.data() + f * nSamples;
      float mean = std::accumulate(begin, begin + nSamples, 0.0);
      mean /= nSamples;
      std::transform(
          begin, begin + nSamples, begin, [mean](float x) { return x - mean; });
    }
  }
  if (featParams_.preemCoef != 0) {
    std::cout << "PREEM RAN";
    std::cout << featParams_.preemCoef;
    preEmphasis_.applyInPlace(frames);
  }


  // 2. after preEmphasis
  for(int i=0;i<frames.size();i++)
  {
      file<<frames[i]<<std::endl;
  }
  file << std::endl;


  
  windowing_.applyInPlace(frames);

  // 3. after windowing
  for(int i=0;i<frames.size();i++)
  {
      file<<frames[i]<<std::endl;
  }
  file << std::endl;


  std::vector<float> dft(K * nFrames);
  for (size_t f = 0; f < nFrames; ++f) {
    auto begin = frames.data() + f * nSamples;
    {
      std::lock_guard<std::mutex> lock(fftMutex_);
      std::copy(begin, begin + nSamples, inFftBuf_.data());
      std::fill(outFftBuf_.begin(), outFftBuf_.end(), 0.0);
      fftw_execute(*fftPlan_);

      // Copy stuff to the redundant part
      for (size_t i = K; i < nFft; ++i) {
        outFftBuf_[2 * i] = outFftBuf_[2 * nFft - 2 * i];
        outFftBuf_[2 * i + 1] = -outFftBuf_[2 * nFft - 2 * i + 1];
      }

      for (size_t i = 0; i < K; ++i) {
        dft[f * K + i] = std::sqrt(
            outFftBuf_[2 * i] * outFftBuf_[2 * i] +
            outFftBuf_[2 * i + 1] * outFftBuf_[2 * i + 1]);
      }
    }

    // 3. print dft
    for(int i=0;i<dft.size();i++)
    {
        file<<dft[i]<<std::endl;
    }
    file << std::endl;


  }

  file.close();

  // 4. print dft
  for(int i=0;i<dft.size();i++)
  {
      file<<dft[i]<<std::endl;
  }
  file << std::endl;


  return dft;
}

std::vector<float> PowerSpectrum::batchApply(
    const std::vector<float>& input,
    int batchSz) {
  if (batchSz <= 0) {
    throw std::invalid_argument("PowerSpectrum: negative batchSz");
  } else if (input.size() % batchSz != 0) {
    throw std::invalid_argument(
        "PowerSpectrum: input size is not divisible by batchSz");
  }
  int N = input.size() / batchSz;
  int outputSz = outputSize(N);
  std::vector<float> feat(outputSz * batchSz);

#pragma omp parallel for num_threads(batchSz)
  for (int b = 0; b < batchSz; ++b) {
    auto start = input.begin() + b * N;
    std::vector<float> inputBuf(start, start + N);
    auto curFeat = apply(inputBuf);
    if (outputSz != curFeat.size()) {
      throw std::logic_error("PowerSpectrum: apply() returned wrong size");
    }
    std::copy(
        curFeat.begin(), curFeat.end(), feat.begin() + b * curFeat.size());
  }
  return feat;
}

FeatureParams PowerSpectrum::getFeatureParams() const {
  return featParams_;
}

int PowerSpectrum::outputSize(int inputSz) {
  return featParams_.powSpecFeatSz() * featParams_.numFrames(inputSz);
}

void PowerSpectrum::validatePowSpecParams() const {
  if (featParams_.samplingFreq <= 0) {
    throw std::invalid_argument("PowerSpectrum: samplingFreq is negative");
  } else if (featParams_.frameSizeMs <= 0) {
    throw std::invalid_argument("PowerSpectrum: frameSizeMs is negative");
  } else if (featParams_.frameStrideMs <= 0) {
    throw std::invalid_argument("PowerSpectrum: frameStrideMs is negative");
  } else if (featParams_.numFrameSizeSamples() <= 0) {
    throw std::invalid_argument("PowerSpectrum: frameSizeMs is too low");
  } else if (featParams_.numFrameStrideSamples() <= 0) {
    throw std::invalid_argument("PowerSpectrum: frameStrideMs is too low");
  }
}

PowerSpectrum::~PowerSpectrum() {
  fftw_destroy_plan(*fftPlan_);
}
} // namespace audio
} // namespace lib
} // namespace fl
