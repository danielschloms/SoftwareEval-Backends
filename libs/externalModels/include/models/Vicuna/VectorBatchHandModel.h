/*
 * Copyright 2022 Chair of EDA, Technical University of Munich
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	 http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

#include "PerformanceModel.h"
#include "VectorConfig.h"

#define PRINT_REG_COMMITS

#ifdef PRINT_REG_COMMITS
#define CPRINT(...) std::printf(__VA_ARGS__)
#else
#define CPRINT(...)
#endif

namespace Vicuna {

class VectorBatchHandModel : public ConnectorModel {
private:
  static constexpr auto pipeline_depth_ALU = 5;
  static constexpr auto pipeline_depth_MUL = 6;
  static constexpr auto pipeline_depth_ELM = 5;
  static constexpr auto pipeline_depth_LSU = 5;

  // Vector register timestamps
  // std::array<uint64_t, 32> vectorRegisterReads{0};
  std::array<uint64_t, 32> vectorRegisterWrites{0};

  // Vector signal timestamps
  uint64_t vsetSignal = 0;
  uint64_t wbFreeSignal = 0;
  uint64_t memArbiterSignal = 0;

  // Dispatch OK times
  uint64_t dispatch_1_ready = 0;
  uint64_t dispatch_2_ready = 0;

  auto getVs1Index() -> uint64_t { return vs1_ptr[getInstrIndex()]; }

  auto getVs2Index() -> uint64_t { return vs2_ptr[getInstrIndex()]; }

  auto getVs3Index() -> uint64_t { return vs3_ptr[getInstrIndex()]; }

  auto getVdIndex() -> uint64_t { return vd_ptr[getInstrIndex()]; }

  auto getSew() -> uint64_t {
    uint64_t const vtype = vtype_ptr[getInstrIndex()];
    uint64_t const vsew = (vtype >> 3) & 0b11;
    // SEW can be calculated by shifting 8 left by the register value (vsew)
    return 8 << vsew;
  }

  auto log2(uint64_t value) -> uint64_t {
    auto result = 0;
    value >>= 1;
    while (value) {
      result++;
      value >>= 1;
    }
    return result;
  }

  auto decodeMultiplicativeLmul(uint64_t const encodedLmul) -> uint64_t {
    static constexpr auto lmulValueBitmask = 0b11;
    return 1 << (encodedLmul & lmulValueBitmask);
  }

  auto getEncodedLmul() -> uint64_t {
    uint64_t const vtype = vtype_ptr[getInstrIndex()];
    static constexpr auto vlmulBitmask = 0b111;
    return vtype & vlmulBitmask;
  }

  auto lmulIsFractional(uint64_t const lmul) -> bool {
    static constexpr auto fractionalLmulBitmask = 0b100;
    return (lmul & fractionalLmulBitmask) != 0;
  }

  auto lmulIsFractional() -> bool {
    uint64_t const vtype = vtype_ptr[getInstrIndex()];
    static constexpr auto fractionalLmulBitmask = 0b100;
    auto const isFractionalLmul = vtype & fractionalLmulBitmask;
    return isFractionalLmul != 0;
  }

public:
  VectorBatchHandModel(PerformanceModel *parent_)
      : ConnectorModel("VectorBatchHandModel", parent_){
            // vlen_ = std::stoi(std::getenv("VLEN"));
            // vlane_width_ = std::stoi(std::getenv("VLANE_WIDTH"));
            // if (vlane_width_ * 2 >= vlen_) {
            //   unpack_2_depth--;
            // }
            // if (VectorConfig::vMemWidth * 2 >= vlen_) {
            //   unpack_1_depth--;
            // }
        };

  uint64_t *vs1_ptr;
  uint64_t *vs2_ptr;
  uint64_t *vs3_ptr;
  uint64_t *vd_ptr;
  uint64_t *vm_ptr;
  uint64_t *vtype_ptr;

  // Loads
  uint64_t *lsWidth_ptr;

  // Whole register loads
  uint64_t *nf_ptr;

  auto getLmul() -> uint64_t {
    uint64_t const vtype = vtype_ptr[getInstrIndex()];
    static constexpr auto fractionalLmulBitmask = 0b100;
    auto const isFractionalLmul = vtype & fractionalLmulBitmask;
    auto lmul = 1;

    if (!isFractionalLmul && ((vtype & 0b11) != 0)) {
      static constexpr auto lmulValueBitmask = 0b11;
      lmul = 1 << (vtype & lmulValueBitmask);
    }

    return lmul;
  }

  constexpr auto one() -> uint64_t { return 1; }

  auto getLoadStoreEmul() -> uint64_t {
    auto const loadWidth = lsWidth_ptr[getInstrIndex()];
    auto const sew = getSew();

    // EMUL = LMUL >= m1 if loadWidth == sew
    auto emul = getLmul();

    if (loadWidth > sew) {
      // EMUL = multiple of LMUL
      // Calculate by adding log2(loadWidth / sew) to vlmul
      // If the fractional bit is still 1, EMUL = 1
      // otherwise just the decoded multiplicative LMUL of that new value.
      // m8 overflowing should result in an illegal instruction anyway, so it is
      // not checked. See spec.
      auto const encodedEmul = getEncodedLmul() + log2(loadWidth / sew);
      emul = lmulIsFractional(encodedEmul)
                 ? 1
                 : decodeMultiplicativeLmul(encodedEmul);
    } else if (loadWidth < sew) {
      // EMUL = fraction of LMUL
      // If LMUL fractional, or log2(loadWidth / sew) >= vlmul, EMUL = 1
      // otherwise subtract log2(loadWidth / sew) from vlmul and decode.
      // See spec.
      auto const vlmul = getEncodedLmul();
      auto const decrement = log2(sew / loadWidth);
      if (lmulIsFractional() || decrement > vlmul) {
        emul = 1;
      } else {
        emul = decodeMultiplicativeLmul(vlmul - decrement);
      }
    }
    return emul;
  }

  // NF = encodedNF + 1
  auto getNf() -> uint64_t { return nf_ptr[getInstrIndex()] + 1; }

  auto getVs1() -> uint64_t { return vectorRegisterWrites[getVs1Index()]; }
  auto getVs2() -> uint64_t { return vectorRegisterWrites[getVs2Index()]; }
  auto getVs3() -> uint64_t { return vectorRegisterWrites[getVs3Index()]; }
  auto getVs1(unsigned lmulIndex) -> uint64_t {
    return vectorRegisterWrites[getVs1Index() + lmulIndex] + 1;
  }
  auto getVs2(unsigned lmulIndex) -> uint64_t {
    return vectorRegisterWrites[getVs2Index() + lmulIndex] + 1;
  }
  auto getVs3(unsigned lmulIndex) -> uint64_t {
    return vectorRegisterWrites[getVs3Index() + lmulIndex] + 1;
  }

  auto setVd(uint64_t timestamp) {
    vectorRegisterWrites[getVdIndex()] = timestamp;
  }
  auto setVd(unsigned lmulIndex, uint64_t timestamp) {
    vectorRegisterWrites[getVdIndex() + lmulIndex] = timestamp;
  }

  auto getWbFreeSignal(void) -> uint64_t { return wbFreeSignal; }
  auto setWbFreeSignal(uint64_t timestamp) -> void { wbFreeSignal = timestamp + 1; }

  void setVsetSignal(uint64_t timestamp) { vsetSignal = timestamp - 1; }
  auto getVsetSignal() -> uint64_t { return vsetSignal; };

  auto setMemArbiterSignal(uint64_t timestamp) -> void {
    memArbiterSignal = timestamp;
  }
  auto getMemArbiterSignal() -> uint64_t { return memArbiterSignal; };

  // Get the max. timestamp for a target register group
  uint64_t getMaxVdGroup(void) {
    auto const emul = getLmul();
    auto const registerBaseIndex = getVdIndex();
    auto start = vectorRegisterWrites.begin() + registerBaseIndex;
    // Ready 1 cycle after register done
    return std::max((*(std::max_element(start, start + emul))) + 1,
                    dispatch_2_ready);
  }

  uint64_t getMaxVdGroupWide(void) {
    auto const emul = 2 * getLmul();
    auto const registerBaseIndex = getVdIndex();
    auto start = vectorRegisterWrites.begin() + registerBaseIndex;
    // Ready 1 cycle after register done
    return (*(std::max_element(start, start + emul))) + 1;
  }

  // Get the max. timestamp for a register group based on nf
  uint64_t getMaxVdGroupNf(void) {
    auto const nFields = getNf();
    auto const registerBaseIndex = getVdIndex();
    auto start = vectorRegisterWrites.begin() + registerBaseIndex;
    // Ready 1 cycle after register done
    return std::max((*(std::max_element(start, start + nFields))) + 1,
                    dispatch_1_ready);
  }

  uint64_t getDispatch1Ready(void) { return dispatch_1_ready; }

  uint64_t getDispatch2Ready(void) { return dispatch_2_ready; }

  // Get the max. timestamp for a register group for load instructions
  uint64_t getMaxVdGroupLoad(void) {
    auto const emul = getLoadStoreEmul();
    auto const registerBaseIndex = getVdIndex();
    auto start = vectorRegisterWrites.begin() + registerBaseIndex;
    // Ready 1 cycle after register done
    return std::max((*(std::max_element(start, start + emul))) + 1,
                    dispatch_1_ready);
  }

  // auto getWaitTime(uint64_t emul) -> uint64_t {
  //   return emul * (vlen_ / vlane_width_);
  // }

  // auto getWaitTimeLsu(uint64_t emul) -> uint64_t {
  //   return emul * (vlen_ / VectorConfig::vMemWidth);
  // }
};

} // namespace Vicuna