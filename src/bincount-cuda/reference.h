template <typename input_t, typename IndexType>
static IndexType
get_bin(input_t v, input_t minvalue, input_t maxvalue, IndexType nbins)
{
  IndexType bin = (v - minvalue) * nbins / (maxvalue - minvalue);
  if (bin == nbins) bin--;
  return bin;
}

template <typename output_t, typename input_t, typename IndexType>
void reference (
       output_t *output,
  const input_t *input,
  IndexType nbins,
  input_t minvalue,
  input_t maxvalue,
  IndexType input_size,
  IndexType output_size,
  const int repeat)
{
  for (int n = 0; n < repeat; n++) {
    // compute histogram for the block
    for (IndexType i = 0; i < input_size; i++) {
      const auto v = input[i];
      if (v >= minvalue && v <= maxvalue) {
        const IndexType bin = get_bin<input_t, IndexType>(
                              v, minvalue, maxvalue, nbins);
        output[bin] += 1;
      }
    }
  }
}
