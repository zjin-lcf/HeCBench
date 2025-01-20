#pragma once

#include <sycl/sycl.hpp>
#include <limits>
#include <map>
#include <string>
#include "atomics.h"

enum ReductionType { SUM, MUL, DIV, MIN, MAX };

const std::map<std::string, ReductionType> reduce2REDUCE = {
    {"sum", SUM}, {"mul", MUL}, {"div", DIV}, {"min", MIN}, {"max", MAX}
};

template <typename scalar_t, ReductionType REDUCE> struct Reducer {
  static inline scalar_t init() {
    if (REDUCE == MUL || REDUCE == DIV)
      return (scalar_t)1;
    else if (REDUCE == MIN)
      return std::numeric_limits<scalar_t>::max();
    else if (REDUCE == MAX)
      return std::numeric_limits<scalar_t>::lowest();
    else
      return (scalar_t)0;
  }

  static inline void update(scalar_t *val, scalar_t new_val) {
    if (REDUCE == SUM)
      *val = *val + new_val;
    else if (REDUCE == MUL)
      *val = *val * new_val;
    else if (REDUCE == DIV)
      *val = *val / new_val;
    else if ((REDUCE == MIN && new_val < *val) ||
             (REDUCE == MAX && new_val > *val)) {
      *val = new_val;
    }
  }

  static inline void atomic_write(scalar_t *address, scalar_t val) {
    if (REDUCE == SUM)
      atomicAdd(address, val);
    else if (REDUCE == MUL)
      atomicMul(address, val);
    else if (REDUCE == DIV)
      atomicDiv(address, val);
    else if (REDUCE == MIN)
      atomicMin(address, val);
    else if (REDUCE == MAX)
      atomicMax(address, val);
  }
};
