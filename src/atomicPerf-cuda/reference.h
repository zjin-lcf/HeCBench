template <typename T>
void BlockRangeAtomicOnGlobalMem_ref(T* data, int n)
{
  for (int i = 0; i < n; i++) {
    data[i % BLOCK_SIZE] += (T)1;
  }
}

template <typename T>
void WarpRangeAtomicOnGlobalMem_ref(T* data, int n)
{
  for (int i = 0; i < n; i++) {
    data[i & 0x1F] += (T)1;
  }
}

template <typename T>
void SingleRangeAtomicOnGlobalMem_ref(T* data, int offset, int n)
{
  for (int i = 0; i < n; i++) {
    data[offset] += (T)1;
  }
}
