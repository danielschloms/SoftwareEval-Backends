#pragma once

#include <cstdint>
#include <cstdlib>

#include "PerformanceModel.h"

namespace Vicuna {

class VectorDivDelay : public ResourceModel {
public:
  VectorDivDelay(PerformanceModel *parent_)
      : ResourceModel("VectorDivDelay", parent_) {
    vlen_ = std::stoi(std::getenv("VLEN"));
    vlane_width_ = std::stoi(std::getenv("VLANE_WIDTH"));
  };

  uint64_t *vtype_ptr;

  int getDelay(void) {
    constexpr auto dividerWidth = 32;
    constexpr auto dividerCycles = 35;
    auto const nParallelDivisions = vlane_width_ / dividerWidth;
    auto const nElements = vlen_ / getSew();
    auto const nDivisions = nElements / nParallelDivisions;
    auto const cyclesPerRegister = nDivisions * dividerCycles;
    return cyclesPerRegister;
  }

private:
  uint64_t vlen_;
  uint64_t vlane_width_;
  auto getSew() -> uint64_t {
    uint64_t const vtype = vtype_ptr[getInstrIndex()];
    uint64_t const vsew = (vtype >> 3) & 0b11;
    // SEW can be calculated by shifting 8 left by the register value (vsew)
    return 8 << vsew;
  }
};

} // namespace Vicuna
