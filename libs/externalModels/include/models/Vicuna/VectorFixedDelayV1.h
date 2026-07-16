#pragma once

#include <cstdint>
#include <cstdlib>

#include "PerformanceModel.h"

namespace Vicuna {

class VectorFixedDelayV1 : public ResourceModel {
public:
  VectorFixedDelayV1(PerformanceModel *parent_)
      : ResourceModel("VectorFixedDelayV1", parent_) {
    vlen_ = std::stoi(std::getenv("VLEN"));
    vlane_width_ = 32;
  };

  int getDelay(void) { return vlen_ / vlane_width_; }

private:
  uint64_t vlen_;
  uint64_t vlane_width_;
};

} // namespace Vicuna
