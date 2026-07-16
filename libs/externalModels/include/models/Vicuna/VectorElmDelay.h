#pragma once

#include <cstdint>
#include <cstdlib>

#include "PerformanceModel.h"

namespace Vicuna {

class VectorElmDelay : public ResourceModel {
public:
  VectorElmDelay(PerformanceModel *parent_)
      : ResourceModel("VectorElmDelay", parent_) {
    vlen_ = std::stoi(std::getenv("VLEN"));
  };

  uint64_t *vtype_ptr;

  int getDelay(void) { return vlen_ / getSew(); }

private:
  uint64_t vlen_;
  auto getSew() -> uint64_t {
    uint64_t const vtype = vtype_ptr[getInstrIndex()];
    uint64_t const vsew = (vtype >> 3) & 0b11;
    // SEW can be calculated by shifting 8 left by the register value (vsew)
    return 8 << vsew;
  }
};

} // namespace Vicuna
