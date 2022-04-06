// 
// Copyright 2004-present Facebook. All Rights Reserved.
//

#include <CL/sycl.hpp>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <vector>

namespace facebook { namespace cuda {

// test the warp-wide sort code
std::vector<float> sort(sycl::queue &q, const std::vector<float>& data);

// test the warp-wide sort with indices code
std::vector<std::pair<float, int> >
sortWithIndices(sycl::queue &q, const std::vector<float>& data);

} } // namespace

// Add in +/- inf, +/- 0, denorms
void addSpecialFloats(std::vector<float>& vals) {
  // Add in +/- infinity, with duplicates
  vals.push_back(std::numeric_limits<float>::infinity());
  vals.push_back(std::numeric_limits<float>::infinity());
  vals.push_back(-std::numeric_limits<float>::infinity());
  vals.push_back(-std::numeric_limits<float>::infinity());

  // Add in +/- zero, with duplicates
  vals.push_back(0.0f);
  vals.push_back(0.0f);
  vals.push_back(-0.0f);
  vals.push_back(-0.0f);

  // Add in some denorm floats, with duplicates
  vals.push_back(std::numeric_limits<float>::denorm_min() * 4.0f);
  vals.push_back(std::numeric_limits<float>::denorm_min());
  vals.push_back(std::numeric_limits<float>::denorm_min());
  vals.push_back(-std::numeric_limits<float>::denorm_min());
  vals.push_back(-std::numeric_limits<float>::denorm_min());
  vals.push_back(-std::numeric_limits<float>::denorm_min() * 4.0f);
}


bool test_sort(sycl::queue &q) {
  std::vector<float> vals;
  addSpecialFloats(vals);

  std::vector<float> sorted = vals;
  std::sort(sorted.begin(), sorted.end(), std::greater<float>());

  bool ok = true;
  for (int i = 0; i < 3; ++i) {
    std::shuffle(vals.begin(), vals.end(), std::random_device());
    auto out = facebook::cuda::sort(q, vals);

    if (sorted.size() != out.size()) {
      ok = false;
      goto DONE;
    }

    for (int j = 0; j < (int)out.size(); ++j) {
      if (sorted[j] != out[j]) {
        ok = false;
        goto DONE;
      }
    }
  }
  DONE:
  return ok;
}

bool test_sortInRegisters(sycl::queue &q) {
  // Test sorting std::vectors of size 1 to 4 x warpSize, which is the
  // maximum in-register size we support
  bool ok = true;

  for (int size = 1; size <= 4 * 32; ++size) {
    std::vector<float> vals;

    for (int i = 0; i < size; ++i) {
      vals.push_back((float) i + 1);
    }

    std::vector<float> sorted = vals;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());

    for (int i = 0; i < 3; ++i) {
      std::shuffle(vals.begin(), vals.end(), std::random_device());
      auto out = facebook::cuda::sort(q, vals);

      if (sorted.size() != out.size()) {
        ok = false;
        goto DONE;
      }

      for (int j = 0; j < (int)out.size(); ++j) {
        if (sorted[j] != out[j]) {
          ok = false;
          goto DONE;
        }
      }
    }
  }
  DONE:
  return ok;
}

bool test_sortIndicesInRegisters(sycl::queue &q) {
  // Test sorting std::vectors of size 1 to 4 x warpSize, which is the
  // maximum in-register size we support

  bool ok = true;
  for (int size = 1; size <= 4 * 32; ++size) {
    std::vector<float> vals;

    for (int i = 0; i < size; ++i) {
      vals.push_back((float) i);
    }

    std::vector<float> sorted = vals;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());

    for (int i = 0; i < 3; ++i) {
      std::shuffle(vals.begin(), vals.end(), std::random_device());
      auto out = facebook::cuda::sortWithIndices(q, vals);

      if (sorted.size() != out.size()) {
        ok = false;
        goto DONE;
      }

      for (int j = 0; j < (int)out.size(); ++j) {
        if (sorted[j] != out[j].first) {
          ok = false;
          goto DONE;
        }

        int idx = out[j].second;
        if (idx < 0 || idx >= (int)vals.size()  || out[j].first != vals[idx]) {
          ok = false;
          goto DONE;
        }
      }

      // Test for uniqueness of indices
      std::unordered_set<int> indices;
      for (const auto &p : out) {
        if (indices.count(p.second) == true) {
          ok = false;
          goto DONE;
        }
        indices.emplace(p.second);
      }
    }
  }
  DONE:
  return ok;
}

int main(int argc, char** argv) {
#ifdef USE_GPU
  sycl::gpu_selector dev_sel;
#else
  sycl::cpu_selector dev_sel;
#endif
  sycl::queue q(dev_sel);

  bool ok;
  ok = test_sort(q);
  printf("%s: test_sort\n", ok ? "PASS" : "FAIL");

  ok = test_sortInRegisters(q);
  printf("%s: test_sortInRegisters\n", ok ? "PASS" : "FAIL");

  ok = test_sortIndicesInRegisters(q);
  printf("%s: test_sortIndicesInRegisters\n", ok ? "PASS" : "FAIL");

  return 0;
}
