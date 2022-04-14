/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <stdexcept>
#include <string> // header file for string
// #include <experimental/filesystem>

#include "flashlight/fl/nn/modules/Module.h"
#include "flashlight/fl/common/Utils.h"
#include "flashlight/fl/nn/Init.h"

// namespace fs = std::experimental::filesystem;
namespace fl {

Module::Module() = default;

Module::Module(const std::vector<Variable>& params)
    : params_(params.begin(), params.end()) {}

Variable Module::param(int position) const {
  if (!(position >= 0 && position < params_.size())) {
    throw std::out_of_range("Module param index out of range");
  }
  return params_[position];
}

void Module::setParams(const Variable& var, int position) {
  if (!(position >= 0 && position < params_.size())) {
    throw std::out_of_range("Module param index out of range");
  }
  params_[position] = var;
}

void Module::train() {
  train_ = true;
  for (auto& param : params_) {
    param.setCalcGrad(true);
  }
}

void Module::zeroGrad() {
  for (auto& param : params_) {
    param.zeroGrad();
  }
}

void Module::eval() {
  train_ = false;
  for (auto& param : params_) {
    param.setCalcGrad(false);
  }
}

std::vector<Variable> Module::params() const {
  return params_;
}

std::vector<Variable> Module::operator()(const std::vector<Variable>& input) {
  return this->forward(input);
}

int Module::paramSize() const {
  return params_.size();
}

std::string Module::printWeights(std::string path) const {
  std::ostringstream ss;
  ss << "\nModule[0]::printWeights start\n";
  // ss << this->prettyString();
  // ss << modules_[0]->printWeights(path);
  ss << modules_[0]->prettyString();
  ss << "Module[0]::prettyString() done\n";
  // this->printWeights(path);
  return ss.str();
}

// std::string Module::printWeights(std::string path) const {
//   int start = 0;
//   std::ostringstream ss;
  
//   std::string prettyString = this->prettyString();
//   std::string delimiter = " ";
//   std::string moduleName = prettyString.substr(0, prettyString.find(delimiter));

//   ss << "Module::printWeights start\n";
//   ss << prettyString;
//   ss << "Module::prettyString() done\n";
//   // fs::path dir (path);
//   // fs::path file (moduleName + ".arr");
//   // fs::path full_path = dir / file;
//   // const char* full_path_char = full_path.string().c_str();
//   std::string fileName = moduleName + ".arr";
//   std::string full_path = path + fileName;
//   const char* full_path_char = full_path.c_str();
//   ss << "Saving to: " << full_path_char;
  
//   for (int i = 0; i < params_.size(); i++) {
//     std::string paramname = "param_" + std::to_string(start + i);
//     // append = True
//     af::saveArray(paramname.c_str(), params_[i].array(), full_path_char, true); 
//     ss << "\n" << paramname;
//     // ss << af::toString("weights", params_[i].array());
//   }
  
//   return ss.str();
// }


// std::string Module::printWeights(std::string path) const {
//   int start = 0;
//   std::ostringstream ss;
  
//   std::string prettyString = this->prettyString();
//   std::string delimiter = " ";
//   std::string moduleName = prettyString.substr(0, prettyString.find(delimiter));
//   ss << "Module::printWeights start\n";
//   ss << prettyString;
//   ss << "Module::prettyString() done\n";
//   // fs::path dir (path);
//   // fs::path file (moduleName + ".arr");
//   // fs::path full_path = dir / file;
//   // const char* full_path_char = full_path.string().c_str();
//   std::string fileName = moduleName + ".arr";
//   std::string full_path = path + fileName;
//   const char* full_path_char = full_path.c_str();
//   ss << "Saving to: " << full_path_char;
  
//   for (int i = 0; i < params_.size(); i++) {
//     std::string paramname = "param_" + std::to_string(start + i);
//     // append = True
//     af::saveArray(paramname.c_str(), params_[i].array(), full_path_char, true); 
//     ss << "\n" << paramname;
//     // ss << af::toString("weights", params_[i].array());
//   }
  
//   return ss.str();
// }


// std::string Module::printWeights(std::string path, int start = 0) const {
//   std::ostringstream ss;
//   ss << this->prettyString();

//   for (int i = 0; i < params_.size(); i++) {
//     std::string paramname = "param_" + std::to_string(start + i);
//     std::string temp = path + paramname + ".arr";
//     af::saveArray(paramname.c_str(), params_[i].array(), temp.c_str());
//     ss << paramname + "\n";
//     // ss << af::toString("weights", params_[i].array());
//   }
  
//   return ss.str();
// }


UnaryModule::UnaryModule() = default;

UnaryModule::UnaryModule(const std::vector<Variable>& params)
    : Module(params) {}

std::vector<Variable> UnaryModule::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("UnaryModule expects only one input");
  }
  return {forward(inputs[0])};
}

Variable UnaryModule::operator()(const Variable& input) {
  return this->forward(input);
}

BinaryModule::BinaryModule() = default;

BinaryModule::BinaryModule(const std::vector<Variable>& params)
    : Module(params) {}

std::vector<Variable> BinaryModule::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() != 2) {
    throw std::invalid_argument("BinaryModule expects two inputs");
  }
  return {forward(inputs[0], inputs[1])};
}

Variable BinaryModule::operator()(
    const Variable& input1,
    const Variable& input2) {
  return this->forward(input1, input2);
}






} // namespace fl
