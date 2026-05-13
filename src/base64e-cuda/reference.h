// "expected_output" Host processing
int reference(const void* data_buf, size_t dataLength, uchar* result, size_t resultSize, size_t *encSize)
{
  const uchar base64chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const uchar *data = (const uchar *)data_buf;
  size_t resultIndex = 0;
  size_t x;
  unsigned int n = 0;
  int padCount = dataLength % 3;
  uchar n0, n1, n2, n3;

  /* increment over the length of the string, three characters at a time */
  for (x = 0; x < dataLength; x += 3)
  {
    /* these three 8-bit (ASCII) characters become one 24-bit number */
    n = ((unsigned int)data[x]) << 16; //parenthesis needed, compiler depending on flags can do the shifting before conversion to uint32_t, resulting to 0

    if((x+1) < dataLength)
      n += ((unsigned int)data[x+1]) << 8;//parenthesis needed, compiler depending on flags can do the shifting before conversion to uint32_t, resulting to 0

    if((x+2) < dataLength)
      n += data[x+2];

    /* this 24-bit number gets separated into four 6-bit numbers */
    n0 = (uchar)(n >> 18) & 63;
    n1 = (uchar)(n >> 12) & 63;
    n2 = (uchar)(n >> 6) & 63;
    n3 = (uchar)n & 63;

    /*
     * if we have one byte available, then its encoding is spread
     * out over two characters
     */
    if(resultIndex >= resultSize) return 1;   /* indicate failure: buffer too small */
    result[resultIndex++] = base64chars[n0];
    if(resultIndex >= resultSize) return 1;   /* indicate failure: buffer too small */
    result[resultIndex++] = base64chars[n1];

    /*
     * if we have only two bytes available, then their encoding is
     * spread out over three chars
     */
    if((x+1) < dataLength)
    {
      if(resultIndex >= resultSize) return 1;   /* indicate failure: buffer too small */
      result[resultIndex++] = base64chars[n2];
    }

    /*
     * if we have all three bytes available, then their encoding is spread
     * out over four characters
     */
    if((x+2) < dataLength)
    {
      if(resultIndex >= resultSize) return 1;   /* indicate failure: buffer too small */
      result[resultIndex++] = base64chars[n3];
    }
  }

  /*
   * create and add padding that is required if we did not have a multiple of 3
   * number of characters available
   */
  if (padCount > 0)
  {
    for (; padCount < 3; padCount++)
    {
      if(resultIndex >= resultSize) return 1;   /* indicate failure: buffer too small */
      result[resultIndex++] = '=';
    }
  }

  if(resultIndex >= resultSize) return 1;   /* indicate failure: buffer too small */
  result[resultIndex] = 0;
  *encSize = resultIndex;
  return 0;   /* indicate success */
}

