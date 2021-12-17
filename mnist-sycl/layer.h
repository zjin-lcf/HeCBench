#ifndef LAYER_H
#define LAYER_H
#endif

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

class Layer {
  public:
    int M, N, O;

    buffer<float, 1> output;
    buffer<float, 1> preact;

    buffer<float, 1> bias;
    buffer<float, 1> weight;

    buffer<float, 1> d_output;
    buffer<float, 1> d_preact;
    buffer<float, 1> d_weight;

    Layer(queue &q, int M, int N, int O);

    void setOutput(queue &q, float *data);
    void clear(queue &q);
    void bp_clear(queue &q);
};
