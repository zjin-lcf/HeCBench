# MULTIANS - Massively Parallel ANS Decoding on GPUs

An implementation of a novel algorithm for ANS (Asymmetric Numeral Systems) decoding on GPUs.

For a detailed description of the concept, please refer to our [conference paper](https://doi.org/10.1145/3337821.3337888).

The algorithm is capable of decoding raw (unpartitioned) ANS-encoded datastreams of variable size at extremely high throughput rates.

State count and alphabet size are configurable. At its current increment, the decoder supports input data encoded using a single table and a radix of `b = 2` (i.e. encoder emits single bits during renormalization), and alphabet sizes of up to `256` symbols. Another implementation supporting multiple tables / multiple states is subject of future work.

## Requirements

* OpenMP target offloading support
* GNU/Linux
* latest proprietary graphics drivers

### Test program

The test program will generate multiple random datasets (256 symbols) of user-specified size. The symbols are exponentially distributed with increasing rate parameters (λ), yielding different compression ratios for different sets.

For each dataset, the program will:

1. encode the data into a single compressed stream using tANS
2. copy / decode the compressed data on a specified GPU
3. verify the decoded stream

#### Compiling the test program

To compile the test program, configure the Makefile as described above. For example:
`mkdir bin`  

`make -f Makefile.aomp`

#### Running the test program

`./bin/demo <size of input in megabytes>`
