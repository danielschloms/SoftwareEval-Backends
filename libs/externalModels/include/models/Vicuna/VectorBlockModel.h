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
#include <optional>
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

class VectorBlockModel : public ConnectorModel {
private:
  uint64_t vlen_ = 0;
  uint64_t vlane_width_ = 0;
  uint64_t unpack_1_depth = 3;
  uint64_t unpack_2_depth = 3;

  // Vector register timestamps
  std::array<uint64_t, 32> vectorRegisterReads{0};
  std::array<uint64_t, 32> vectorRegisterWrites{0};

  // Vector control timestamps
  // uint64_t timestamp = 0;

  // Vector signal timestamps
  uint64_t vsetSignal = 0;
  uint64_t wbFreeSignal = 0;
  uint64_t memArbiterSignal = 0;

  // Dispatch OK times
  uint64_t dispatch_1_ready = 0;
  uint64_t dispatch_2_ready = 0;

  auto getVs1() -> uint64_t { return vs1_ptr[getInstrIndex()]; }

  auto getVs2() -> uint64_t { return vs2_ptr[getInstrIndex()]; }

  auto getVs3() -> uint64_t { return vs3_ptr[getInstrIndex()]; }

  auto getVd() -> uint64_t { return vd_ptr[getInstrIndex()]; }

  // NF = encodedNF + 1
  auto getNf() -> uint64_t { return nf_ptr[getInstrIndex()] + 1; }

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

public:
  VectorBlockModel(PerformanceModel *parent_)
      : ConnectorModel("VectorBlockModel", parent_) {
    vlen_ = std::stoi(std::getenv("VLEN"));
    vlane_width_ = std::stoi(std::getenv("VLANE_WIDTH"));
    if (vlane_width_ * 2 >= vlen_) {
      unpack_2_depth--;
    }
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

  auto setVBlockDIV_vv(uint64_t timestamp) -> void {
    auto const batchDelay = vlen_ / vlane_width_;
    auto const emul = getLmul();
    auto const vs1 = getVs1();
    auto const vs2 = getVs2();
    auto const vd = getVd();
    static constexpr auto pipeline_depth = 5;
    auto batchStart = timestamp + unpack_2_depth;
    CPRINT("---\nALU\n---\n");
    for (size_t i = 0; i < emul; ++i) {
      auto const maxWriteDependency = std::max(vectorRegisterWrites[vs1 + i],
                                               vectorRegisterWrites[vs2 + i]);
      batchStart =
          std::max(batchStart, maxWriteDependency + 1 + unpack_2_depth);
      CPRINT("Issue @ %lu\n", batchStart);
      vectorRegisterReads[vs1 + i] = batchStart;
      vectorRegisterReads[vs2 + i] = batchStart;
      auto const doneTime =
          std::max(batchStart + batchDelay, vectorRegisterReads[vd + i]);
      vectorRegisterWrites[vd + i] = doneTime + pipeline_depth;
      CPRINT("W v%lu @ %lu\n", vd + i, doneTime + pipeline_depth);
      batchStart = doneTime;
    }
  }

  auto setVBlockDIV_vx(uint64_t timestamp) -> void {
    auto const batchDelay = vlen_ / vlane_width_;
    auto const emul = getLmul();
    auto const vs2 = getVs2();
    auto const vd = getVd();
    static constexpr auto pipeline_depth = 5;
    auto batchStart = timestamp + unpack_2_depth;
    CPRINT("---\nDIV VX\n---\n");
    for (size_t i = 0; i < emul; ++i) {
      batchStart = std::max(batchStart,
                            vectorRegisterWrites[vs2 + i] + unpack_2_depth + 1);
      CPRINT("Issue @ %lu\n", batchStart);
      vectorRegisterReads[vs2 + i] = batchStart;
      auto const doneTime =
          std::max(batchStart + batchDelay, vectorRegisterReads[vd + i]);
      vectorRegisterWrites[vd + i] = doneTime + pipeline_depth;
      CPRINT("W v%lu @ %lu\n", vd + i, doneTime + pipeline_depth);
      batchStart = doneTime;
    }
    dispatch_2_ready = batchStart;
  }

  auto setVBlockALU_vv(uint64_t timestamp) -> void {
    auto const batchDelay = vlen_ / vlane_width_;
    auto const emul = getLmul();
    auto const vs1 = getVs1();
    auto const vs2 = getVs2();
    auto const vd = getVd();
    static constexpr auto pipeline_depth = 5;
    auto batchStart = timestamp + unpack_2_depth;
    // TODO FIX:
    // resolve dependencies in first unpack stage!
    CPRINT("---\nALU\n---\n");
    for (size_t i = 0; i < emul; ++i) {
      auto const maxWriteDependency = std::max(vectorRegisterWrites[vs1 + i],
                                               vectorRegisterWrites[vs2 + i]);
      batchStart =
          std::max(batchStart, maxWriteDependency + 1 + unpack_2_depth);
      CPRINT("Issue @ %lu\n", batchStart);
      vectorRegisterReads[vs1 + i] = batchStart;
      vectorRegisterReads[vs2 + i] = batchStart;
      auto const doneTime =
          std::max(batchStart + batchDelay, vectorRegisterReads[vd + i]);
      vectorRegisterWrites[vd + i] = doneTime + pipeline_depth;
      CPRINT("W v%lu @ %lu\n", vd + i, doneTime + pipeline_depth);
      batchStart = doneTime;
    }
  }

  auto setVBlockALU_vx(uint64_t timestamp) -> void {
    auto const batchDelay = vlen_ / vlane_width_;
    auto const emul = getLmul();
    auto const vs2 = getVs2();
    auto const vd = getVd();
    static constexpr auto pipeline_depth = 5;
    auto batchStart = timestamp + unpack_2_depth;
    CPRINT("---\nALU VX\n---\n");
    for (size_t i = 0; i < emul; ++i) {
      batchStart = std::max(batchStart,
                            vectorRegisterWrites[vs2 + i] + unpack_2_depth + 1);
      CPRINT("Issue @ %lu\n", batchStart);
      vectorRegisterReads[vs2 + i] = batchStart;
      auto const doneTime =
          std::max(batchStart + batchDelay, vectorRegisterReads[vd + i]);
      vectorRegisterWrites[vd + i] = doneTime + pipeline_depth;
      CPRINT("W v%lu @ %lu\n", vd + i, doneTime + pipeline_depth);
      batchStart = doneTime;
    }
    dispatch_2_ready = batchStart - unpack_2_depth - 1;
  }

  auto setVBlockMUL_vv(uint64_t timestamp) -> void {
    auto const batchDelay = vlen_ / vlane_width_;
    auto const emul = getLmul();
    auto const vs1 = getVs1();
    auto const vs2 = getVs2();
    auto const vd = getVd();
    static constexpr auto pipeline_depth = 6;
    auto batchStart = timestamp + unpack_2_depth;
    CPRINT("---\nMUL\n---\n");
    for (size_t i = 0; i < emul; ++i) {
      auto const maxWriteDependency = std::max(vectorRegisterWrites[vs1 + i],
                                               vectorRegisterWrites[vs2 + i]);
      batchStart =
          std::max(batchStart, maxWriteDependency + unpack_2_depth + 1);
      CPRINT("Issue @ %lu\n", batchStart);
      vectorRegisterReads[vs1 + i] = batchStart - 1;
      vectorRegisterReads[vs2 + i] = batchStart - 1;
      auto const doneTime =
          std::max(batchStart + batchDelay, vectorRegisterReads[vd + i]);
      vectorRegisterWrites[vd + i] = doneTime + pipeline_depth;
      CPRINT("W v%lu @ %lu\n", vd + i, doneTime + pipeline_depth);
      batchStart = doneTime;
    }
    dispatch_2_ready = batchStart - unpack_2_depth - 1;
  }

  auto setVBlockMUL_vx(uint64_t timestamp) -> void {
    auto const batchDelay = vlen_ / vlane_width_;
    auto const emul = getLmul();
    auto const vs2 = getVs2();
    auto const vd = getVd();
    static constexpr auto pipeline_depth = 6;
    auto batchStart = timestamp + unpack_2_depth;
    CPRINT("---\nALU VX\n---\n");
    for (size_t i = 0; i < emul; ++i) {
      batchStart = std::max(batchStart,
                            vectorRegisterWrites[vs2 + i] + unpack_2_depth + 1);
      CPRINT("Issue @ %lu\n", batchStart);
      vectorRegisterReads[vs2 + i] = batchStart;
      auto const doneTime =
          std::max(batchStart + batchDelay, vectorRegisterReads[vd + i]);
      vectorRegisterWrites[vd + i] = doneTime + pipeline_depth;
      CPRINT("W v%lu @ %lu\n", vd + i, doneTime + pipeline_depth);
      batchStart = doneTime;
    }
    dispatch_2_ready = batchStart - unpack_2_depth - 1;
  }

  auto getWbFreeSignal(void) -> uint64_t { return wbFreeSignal; }
  auto setWbFreeSignal(uint64_t timestamp) -> void { wbFreeSignal = timestamp; }

  void setVsetSignal(uint64_t timestamp) { vsetSignal = timestamp - 1; }
  auto getVsetSignal() -> uint64_t { return vsetSignal; };

  auto setMemArbiterSignal(uint64_t timestamp) -> void {
    memArbiterSignal = timestamp;
  }
  auto getMemArbiterSignal() -> uint64_t { return memArbiterSignal; };

  auto setVBlockELM_vv(uint64_t timestamp) -> void {
    auto const batchDelay = vlen_ / getSew();
    auto const emul = getLmul();
    auto const vs1 = getVs1();
    auto const vs2 = getVs2();
    auto const vd = getVd();
    auto batchStart = timestamp + unpack_1_depth;
    static constexpr auto pipeline_depth = 5;
    for (size_t i = 0; i < emul; ++i) {
      auto const maxWriteDependency = std::max(vectorRegisterWrites[vs1 + i],
                                               vectorRegisterWrites[vs2 + i]);
      batchStart = std::max(batchStart, maxWriteDependency + 1);
      vectorRegisterReads[vs1 + i] = batchStart - 1;
      vectorRegisterReads[vs2 + i] = batchStart - 1;
      auto const doneTime =
          std::max(batchStart + batchDelay, vectorRegisterReads[vd + i]);
      batchStart = doneTime;
    }

    for (size_t i = 0; i < emul; ++i) {
      vectorRegisterWrites[vd + i] = batchStart + batchDelay + pipeline_depth;
    }
    dispatch_1_ready = batchStart;
  }

  auto setVBlockVMV_V_I(uint64_t timestamp) -> void {
    auto const batchDelay = vlen_ / vlane_width_;
    auto const emul = getLmul();
    auto const vd = getVd();
    static constexpr auto pipeline_depth = 5;
    auto batchStart = timestamp + unpack_2_depth;
    CPRINT("---\nvmv.v.i\n---\n");
    for (size_t i = 0; i < emul; ++i) {
      CPRINT("Issue @ %lu\n", batchStart);
      auto const doneTime =
          std::max(batchStart + batchDelay, vectorRegisterReads[vd + i]);
      vectorRegisterWrites[vd + i] = doneTime + pipeline_depth;
      CPRINT("W v%lu @ %lu\n", vd + i, doneTime + pipeline_depth);
      batchStart = doneTime;
    }
    dispatch_2_ready = batchStart - unpack_2_depth - 1;
  }

  // Scalar -> Vector
  auto setVBlockVMV_S_X(uint64_t timestamp) -> void {
    auto const batchDelay = vlen_ / vlane_width_;
    auto const vd = getVd();
    static constexpr auto pipeline_depth = 5;
    vectorRegisterWrites[vd] =
        timestamp + unpack_2_depth + batchDelay + pipeline_depth;
    dispatch_2_ready = timestamp;
  }

  // Vector -> Scalar
  auto setVBlockVMV_X_S(uint64_t timestamp) -> void {
    auto const batchDelay = 1;
    auto const vs2 = getVs2();
    auto const batchStart = std::max(timestamp, vectorRegisterWrites[vs2] + 1);
    vectorRegisterReads[vs2] = batchStart + unpack_1_depth;
    auto const doneTime = batchStart + unpack_1_depth + batchDelay;
    wbFreeSignal = doneTime;
    dispatch_1_ready = batchStart;
  }

  auto setVBlockStore(uint64_t timestamp) -> void {
    auto const batchDelay = vlen_ / VectorConfig::vMemWidth;
    auto const emul = getLoadStoreEmul();
    auto const vs3 = getVs3();

    auto batchStart = timestamp + unpack_1_depth;
    static constexpr auto pipeline_depth = 5;
    CPRINT("---\nStore\n---\n");
    for (size_t i = 0; i < emul; ++i) {
      // Start batch after last batch (or start of block), or after dependencies
      // are resolved
      batchStart = std::max(batchStart,
                            vectorRegisterWrites[vs3 + i] + unpack_1_depth + 1);
      CPRINT("R v%lu @ %lu\n", vs3 + i, vectorRegisterWrites[vs3 + i]);
      CPRINT("Issue batch @ %lu\n", batchStart);

      // Also update read times for vs3 (parallel execution)
      vectorRegisterReads[vs3 + i] = batchStart;
      // Increment start for next batch by batch delay
      batchStart += batchDelay;
    }
    wbFreeSignal = batchStart + pipeline_depth;
    dispatch_1_ready = batchStart;
    CPRINT("DISP 1 ready @ %lu\n", dispatch_1_ready);
    CPRINT("WB free @ %lu\n", wbFreeSignal);
  }

  auto setVBlockStoreNf(uint64_t timestamp) -> void {
    auto const batchDelay = vlen_ / VectorConfig::vMemWidth;
    auto const emul = getNf();
    auto const vs3 = getVs3();

    auto batchStart = timestamp + unpack_1_depth;
    static constexpr auto pipeline_depth = 5;
    CPRINT("---\nStore\n---\n");
    for (size_t i = 0; i < emul; ++i) {
      // Start batch after last batch (or start of block), or after dependencies
      // are resolved
      batchStart = std::max(batchStart,
                            vectorRegisterWrites[vs3 + i] + unpack_1_depth + 1);
      CPRINT("R v%lu @ %lu\n", vs3 + i, vectorRegisterWrites[vs3 + i]);
      CPRINT("Issue batch @ %lu\n", batchStart);

      // Also update read times for vs3 (parallel execution)
      vectorRegisterReads[vs3 + i] = batchStart;
      // Increment start for next batch by batch delay
      batchStart += batchDelay;
    }
    wbFreeSignal = batchStart + pipeline_depth;
    dispatch_1_ready = batchStart;
    CPRINT("DISP 1 ready @ %lu\n", dispatch_1_ready);
    CPRINT("WB free @ %lu\n", wbFreeSignal);
  }

  // auto setVBlockStoreNf(uint64_t timestamp) -> void {
  //   auto const batchDelay = vlen_ / VectorConfig::vMemWidth;
  //   auto const emul = getNf();
  //   auto const vs3 = getVs3();

  //   auto batchStart = timestamp + 3;
  //   static constexpr auto pipeline_depth = 5;
  //   for (size_t i = 0; i < emul; ++i) {
  //     // Start batch after last batch (or start of block), or after
  //     dependencies
  //     // are resolved
  //     batchStart = std::max(batchStart, vectorRegisterWrites[vs3 + i]);
  //     // Also update read times for vs3 (parallel execution)
  //     vectorRegisterReads[vs3 + i] = batchStart;
  //     // Increment start for next batch by batch delay
  //     batchStart += batchDelay;
  //   }
  //   wbFreeSignal = batchStart + pipeline_depth;
  //   dispatch_1_ready = batchStart;
  // }

  auto setVBlockLoad(uint64_t timestamp) -> void {
    auto const batchDelay = vlen_ / VectorConfig::vMemWidth;
    auto const emul = getLoadStoreEmul();
    auto const vd = getVd();

    static constexpr auto pipeline_depth = 5;
    auto batchStart = timestamp + unpack_1_depth;
    CPRINT("---\nLoad\n---\n");
    for (size_t i = 0; i < emul; ++i) {
      // Update write time for vd, stalls if parallel unit reads from same
      // register
      auto const doneTime =
          std::max(batchStart + batchDelay, vectorRegisterReads[vd + i]);
      vectorRegisterWrites[vd + i] = doneTime + pipeline_depth;
      CPRINT("W v%lu @ %lu\n", vd + i, doneTime + pipeline_depth);
      // Increment start for next batch by batch delay
      batchStart = doneTime;
    }
    // Next instruction can enter while batch still in pipeline
    // But to WB is only clear after writing
    wbFreeSignal = batchStart + pipeline_depth;
    dispatch_1_ready = batchStart;
  }

  auto setVBlockLoadNf(uint64_t timestamp) -> void {
    auto const batchDelay = vlen_ / VectorConfig::vMemWidth;
    auto const emul = getNf();
    auto const vd = getVd();

    static constexpr auto pipeline_depth = 5;
    auto batchStart = timestamp + unpack_1_depth;
    CPRINT("---\nLoad NF\n---\n");
    for (size_t i = 0; i < emul; ++i) {
      // Update write time for vd, stalls if parallel unit reads from same
      // register
      auto const doneTime =
          std::max(batchStart + batchDelay, vectorRegisterReads[vd + i]);
      vectorRegisterWrites[vd + i] = doneTime + pipeline_depth;
      CPRINT("W v%lu @ %lu\n", vd + i, doneTime + pipeline_depth);
      // Increment start for next batch by batch delay
      batchStart = doneTime;
    }
    // Next instruction can enter while batch still in pipeline
    // But to WB is only clear after writing
    wbFreeSignal = batchStart + pipeline_depth;
    dispatch_1_ready = batchStart;
  }

  // Get the max. timestamp for a target register group
  uint64_t getMaxVdGroup(void) {
    auto const emul = getLmul();
    auto const registerBaseIndex = getVd();
    auto start = vectorRegisterWrites.begin() + registerBaseIndex;
    // Ready 1 cycle after register done
    return std::max((*(std::max_element(start, start + emul))) + 1,
                    dispatch_2_ready);
  }

  uint64_t getMaxVdGroupWide(void) {
    auto const emul = 2 * getLmul();
    auto const registerBaseIndex = getVd();
    auto start = vectorRegisterWrites.begin() + registerBaseIndex;
    // Ready 1 cycle after register done
    return (*(std::max_element(start, start + emul))) + 1;
  }

  // Get the max. timestamp for a register group based on nf
  uint64_t getMaxVdGroupNf(void) {
    auto const nFields = getNf();
    auto const registerBaseIndex = getVd();
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
    auto const registerBaseIndex = getVd();
    auto start = vectorRegisterWrites.begin() + registerBaseIndex;
    // Ready 1 cycle after register done
    return std::max((*(std::max_element(start, start + emul))) + 1,
                    dispatch_1_ready);
  }

  auto getWaitTime(uint64_t emul) -> uint64_t {
    return emul * (vlen_ / vlane_width_);
  }

  auto getWaitTimeLsu(uint64_t emul) -> uint64_t {
    return emul * (vlen_ / VectorConfig::vMemWidth);
  }
};

} // namespace Vicuna