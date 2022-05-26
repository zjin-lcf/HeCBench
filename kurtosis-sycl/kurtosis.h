struct kurtosisResult {
  int count;
  float mean;
  float m2;
  float m3;
  float m4;

  kurtosisResult() : count(0),mean(0),m2(0),m3(0),m4(0) {}

  kurtosisResult(int count, float mean, float M2, float M3, float M4) :
    count(count),mean(mean),m2(M2),m3(M3),m4(M4){}
};

template <typename T>
struct kurtosis_unary_op
{
  kurtosisResult operator()(const T& x) const {
    kurtosisResult result;
    result.count = 1;
    result.mean = x.value;
    result.m2 = 0;
    result.m3 = 0;
    result.m4 = 0;
    return result;
  }
};

struct kurtosis_binary_op {
  kurtosisResult operator()(const kurtosisResult& x, const kurtosisResult& y) const {
    float count  = x.count + y.count;
    float count2 = count  * count;
    float count3 = count2 * count;

    float delta  = y.mean - x.mean;
    float delta2 = delta  * delta;
    float delta3 = delta2 * delta;
    float delta4 = delta3 * delta;

    kurtosisResult result;
    result.count = count;

    result.mean = x.mean + delta * y.count / count;

    result.m2  = x.m2 + y.m2;
    result.m2 += delta2 * x.count * y.count / count;

    result.m3  = x.m3 + y.m3;
    result.m3 += delta3 * x.count * y.count * (x.count - y.count) / count2;
    result.m3 += 3.0f * delta * (x.count * y.m2 - y.count * x.m2) / count;

    result.m4  = x.m4 + y.m4;
    result.m4 += delta4 * x.count * y.count * (x.count * x.count - x.count * y.count + y.count * y.count) / count3;
    result.m4 += 6.0f * delta2 * (x.count * x.count * y.m2 + y.count * y.count * x.m2) / count2;
    result.m4 += 4.0f * delta * (x.count * y.m3 - y.count * x.m3) / count;

    return result;
  }
};

size_t kurtosis(sycl::queue &q, storeElement* elements, int elemCount, int repeat, void** result)
{
  kurtosis_unary_op<storeElement> unary_op;
  kurtosis_binary_op binary_op;
  kurtosisResult init;

  auto policy = oneapi::dpl::execution::make_device_policy(q);

  for (int i = 0; i < repeat-1; i++)
    std::transform_reduce(policy,
      elements, elements + elemCount,
      init, binary_op, unary_op);

  kurtosisResult *variance = new kurtosisResult(
    std::transform_reduce(
      policy,
      elements,
      elements + elemCount,
      init, binary_op, unary_op));

  (*result) = variance;

  return sizeof(kurtosisResult);
}
