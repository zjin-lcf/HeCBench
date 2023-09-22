//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
#pragma once

namespace util {

  template <typename T>
    T div_round_up(T val, T divisor) {
      return (val + divisor - 1) / divisor;
    }

  template <typename T>
    T next_multiple(T val, T divisor) {
      return div_round_up(val, divisor) * divisor;
    }
}
