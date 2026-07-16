#pragma once

#include <cstdint>
#include <cstdlib>

#include "PerformanceModel.h"

namespace Vicuna {

class VectorFixedDelayV2 : public ResourceModel {
public:
  VectorFixedDelayV2(PerformanceModel *parent_)
      : ResourceModel("VectorFixedDelayV2", parent_) {
    vlen_ = std::stoi(std::getenv("VLEN"));
    vlane_width_ = std::stoi(std::getenv("VLANE_WIDTH"));
  };

  uint64_t *vtype_ptr;

  int getDelay(void) { return vlen_ / vlane_width_; }

private:
  uint64_t vlen_;
  uint64_t vlane_width_;
};

} // namespace Vicuna
