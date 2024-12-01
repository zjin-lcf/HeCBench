#include <chrono>
#include <fstream>
#include <string>
#include <cmath>
#include <sycl/sycl.hpp>
#include "general.hpp"
#include "kernel.hpp"

// dpct 
#define CHECK_ERROR(expr)                  \
  [&]() {                                  \
    try {                                  \
      expr;                                \
      return 0;                            \
    } catch (std::exception const &e) {    \
      std::cerr << e.what() << std::endl;  \
      return 1;                            \
    }                                      \
  }()


namespace SNICIT_BEY{

class SNICIT{

  private:
    sycl::queue q;
    std::vector<float*> _dev_Y_hidden; // 2 buffers
    std::vector<int*> _dev_hidden_roffw; // hidden layer W row offset
    std::vector<int*> _dev_hidden_colsw; // hidden layer W cols
    std::vector<float*> _dev_hidden_valsw; // hidden layer W vals
    std::vector<float*> _dev_hidden_bias; // hidden layer W vals

    float *Y_input; // on cpu
    float *_dev_Y_output_whole; // on gpu

    float* _dev_Y_input;
    float* _dev_Y_output;
    float* _dev_input_weight;
    float* _dev_input_bias;
    float* _dev_output_weight;
    float* _dev_output_bias;
    int* _dev_result_label;
    
    std::string weight_path, bias_path;
    int num_hidden_neurons, num_layers;
    int input_size, num_classes, num_input, batch_size;
    int nnz;
    float density;
    bool is_cifar;

    // SNICIT augmentation data structures
    int threshold;
    int *y_star_row;
    int seed_size;
    int *centroid_LUT;
    bool *ne_record;
    int *rowsY;

    void _infer();
    
    void _preprocess(const std::string& input_path);
        
    void _weight_bias_alloc_read();

    void _input_alloc_read(const std::string& input_path);

    void _result_alloc_read(const std::string& input_path);

  public:
    SNICIT(
      sycl::queue& _queue,
      const std::string& _weight_path,
      const std::string& _bias_path,
      const int _num_hidden_neurons,
      const int _num_layers,
      const float _density,
      const int _seed_size,
      const int _threshold,
      const int _batch_size,
      const int _num_input,
      const bool _is_cifar
    );

    ~SNICIT();
    
    void infer(const std::string& input_path);

};

SNICIT::SNICIT(
    sycl::queue& _queue,
    const std::string& _weight_path,
    const std::string& _bias_path,
    const int _num_hidden_neurons,
    const int _num_layers,
    const float _density,
    const int _seed_size,
    const int _threshold,
    const int _batch_size,
    const int _num_input,
    const bool _is_cifar
) : q(_queue), weight_path(_weight_path), bias_path(_bias_path), 
    num_hidden_neurons(_num_hidden_neurons), num_layers(_num_layers), 
    num_classes(10), density(_density), 
    nnz(std::round(_num_hidden_neurons*_num_hidden_neurons*_density)), num_input(_num_input), batch_size(_batch_size),
    seed_size(_seed_size), threshold(_threshold), is_cifar(_is_cifar)
 {
  std::cout<<"Constructing SNICIT method......\n";
  input_size = is_cifar ? _num_hidden_neurons : 784;
}

SNICIT::~SNICIT() {
  for(auto& each_Y : _dev_Y_hidden) {
    CHECK_ERROR(sycl::free(each_Y, q));
  }
  for(auto& each_dev_hidden_roffw : _dev_hidden_roffw) {
    CHECK_ERROR(sycl::free(each_dev_hidden_roffw, q));
  }
  for(auto& each_dev_hidden_colsw : _dev_hidden_colsw) {
    CHECK_ERROR(sycl::free(each_dev_hidden_colsw, q));
  }
  for(auto& each_dev_hidden_valsw : _dev_hidden_valsw) {
    CHECK_ERROR(sycl::free(each_dev_hidden_valsw, q));
  }
  for(auto& each_dev_hidden_bias : _dev_hidden_bias) {
    CHECK_ERROR(sycl::free(each_dev_hidden_bias, q));
  }
  if (!is_cifar) {
    CHECK_ERROR(sycl::free(_dev_Y_input, q));
    CHECK_ERROR(sycl::free(_dev_input_weight, q));
    CHECK_ERROR(sycl::free(_dev_input_bias, q));
  }
  CHECK_ERROR(sycl::free(_dev_output_weight, q));
  CHECK_ERROR(sycl::free(_dev_output_bias, q));

  CHECK_ERROR(sycl::free(_dev_Y_output, q));
  CHECK_ERROR(sycl::free(_dev_Y_output_whole, q));
  CHECK_ERROR(sycl::free(_dev_result_label, q));
  CHECK_ERROR(sycl::free(y_star_row, q));
  CHECK_ERROR(sycl::free(centroid_LUT, q));
  CHECK_ERROR(sycl::free(ne_record, q));
  CHECK_ERROR(sycl::free(rowsY, q));

  delete [] Y_input;
}

void SNICIT::infer(const std::string& input_path) {
  _preprocess(input_path);

  _infer();

}

void SNICIT::_preprocess(const std::string &input_path) {
  std::cout<<"preprocessing......\n";
  auto _tic = std::chrono::steady_clock::now();

  _weight_bias_alloc_read();

  _input_alloc_read(input_path);

  _result_alloc_read(input_path);

  CHECK_ERROR(y_star_row = sycl::malloc_shared<int>(seed_size, q));
  CHECK_ERROR(centroid_LUT = sycl::malloc_shared<int>(batch_size, q));
  CHECK_ERROR(ne_record = sycl::malloc_shared<bool>(batch_size, q));
  CHECK_ERROR(rowsY = sycl::malloc_shared<int>(batch_size, q));

  auto _toc = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(_toc - _tic).count();
  std::cout<<"finished preprocessing in "<<duration<< "ms"<<std::endl;
}

void SNICIT::_weight_bias_alloc_read() {
  std::string line;
  std::ifstream MyReadFile;
  int ptr = 0;
  int file_offset;
  if (!is_cifar) {
    file_offset = 2;
    // allocate input layer's weight and bias
    CHECK_ERROR(_dev_input_weight = sycl::malloc_device<float>(
                                   input_size * num_hidden_neurons, q));
    CHECK_ERROR(_dev_input_bias = sycl::malloc_device<float>(
                                   num_hidden_neurons, q));
    // read input layer's weight and bias
    float *input_weight;
    float *input_bias;
    input_weight = new float[input_size * num_hidden_neurons];
    input_bias = new float[num_hidden_neurons];

    MyReadFile = std::ifstream(weight_path+"l1-dense.tsv");
    if (MyReadFile.is_open()) {
      while(std::getline(MyReadFile, line)){
        input_weight[ptr++] = std::stof(line);
      }
    }
    else {
      std::cout << "ERROR: open weight file " << weight_path+"l1-dense.tsv"<<std::endl;
      exit(1);
    }
    MyReadFile.close();
    MyReadFile = std::ifstream(bias_path+"l1-dense.tsv");
    if (MyReadFile.is_open()) {
      ptr = 0;
      while(std::getline(MyReadFile, line)){
        input_bias[ptr++] = std::stof(line);
      }
    }
    else {
      std::cout << "ERROR: open bias file " << weight_path+"l1-dense.tsv"<<std::endl;
      exit(1);
    }
    MyReadFile.close();
    // copy input layer's weight and bias
    CHECK_ERROR(q.memcpy(_dev_input_weight, input_weight,
                     input_size * num_hidden_neurons * sizeof(float)).wait());
    CHECK_ERROR(q.memcpy(_dev_input_bias, input_bias,
                     num_hidden_neurons * sizeof(float)).wait());
    delete [] input_weight;
    delete [] input_bias;
  }
  else {file_offset = 1;}
  for(int hidden_layer = 0; hidden_layer < num_layers; hidden_layer++) {
    int *hidden_roffw;
    int *hidden_colsw;
    float *hidden_valsw;
    float *hidden_bias;

    hidden_roffw = new int[num_hidden_neurons+1];
    hidden_colsw = new int[nnz];
    hidden_valsw = new float[nnz];
    hidden_bias = new float[num_hidden_neurons];
    memset(hidden_roffw, 0, (num_hidden_neurons+1)*sizeof(int));
    int* dev_cur_roffw;
    int* dev_cur_colsw;
    float* dev_cur_valsw;
    float* dev_cur_bias;

    // allocate hidden layer's weight and bias
    CHECK_ERROR(dev_cur_roffw = (int *)sycl::malloc_shared(
                     (num_hidden_neurons + 1) * sizeof(float), q));
    CHECK_ERROR(dev_cur_colsw = (int *)sycl::malloc_shared(
                     nnz * sizeof(float), q));
    CHECK_ERROR(dev_cur_valsw =
                     sycl::malloc_shared<float>(nnz, q));

    // read hidden layer
    
    MyReadFile = std::ifstream(weight_path+"l"+std::to_string(hidden_layer+file_offset)+"-sparse.tsv");
    if (MyReadFile.is_open()) {
      ptr = 0;

      std::vector<std::string> tokens;
      while(std::getline(MyReadFile, line)){

        std::stringstream lineStream(line);
        std::string token;
        tokens.clear();
        while(std::getline(lineStream, token, '\t')) {
          tokens.push_back(std::move(token));
        }

        hidden_roffw[std::stoi(tokens[0])+1]++;
        hidden_colsw[ptr] = std::stoi(tokens[1]);
        hidden_valsw[ptr] = std::stof(tokens[2]);
        ptr++;
      }
      for (int i = 0; i < num_hidden_neurons; i++)
      {
          hidden_roffw[i + 1] += hidden_roffw[i];
      }
    }
    else {
      std::cout << "ERROR: open weight file " << weight_path+"l"+
        std::to_string(hidden_layer+file_offset)+"-sparse.tsv"<<std::endl;
      exit(1);
    }
    MyReadFile.close();
    // copy hidden layer
    CHECK_ERROR(q.memcpy(dev_cur_roffw, hidden_roffw,
                     (num_hidden_neurons + 1) * sizeof(int)).wait());
    CHECK_ERROR(q.memcpy(dev_cur_colsw, hidden_colsw, nnz * sizeof(int)).wait());
    CHECK_ERROR(q.memcpy(dev_cur_valsw, hidden_valsw, nnz * sizeof(float)).wait());
    _dev_hidden_roffw.emplace_back(dev_cur_roffw);
    _dev_hidden_colsw.emplace_back(dev_cur_colsw);
    _dev_hidden_valsw.emplace_back(dev_cur_valsw);

    CHECK_ERROR(dev_cur_bias = sycl::malloc_shared<float>(num_hidden_neurons, q));

    MyReadFile = std::ifstream(bias_path+"l"+std::to_string(hidden_layer+file_offset)+"-sparse.tsv");
    if (MyReadFile.is_open()) {
      ptr = 0;
      while(std::getline(MyReadFile, line)){
        hidden_bias[ptr++] = std::stof(line);
      }
    }
    else {
      std::cout << "ERROR: open bias file " << bias_path+"l"+
        std::to_string(hidden_layer+file_offset)+"-sparse.tsv"<<std::endl;
      exit(1);
    }
    MyReadFile.close();

    CHECK_ERROR(q.memcpy(dev_cur_bias, hidden_bias,
                              num_hidden_neurons * sizeof(float)).wait());

    _dev_hidden_bias.emplace_back(dev_cur_bias);

    delete [] hidden_roffw;
    delete [] hidden_colsw;
    delete [] hidden_valsw;
    delete [] hidden_bias;
  }

  CHECK_ERROR(_dev_output_weight = sycl::malloc_device<float>(
                   num_hidden_neurons * num_classes, q));
  CHECK_ERROR(_dev_output_bias = sycl::malloc_device<float>(num_classes, q));
  float *output_weight;
  float *output_bias;
  output_weight = new float[num_hidden_neurons * num_classes];
  output_bias = new float[num_classes];
  MyReadFile = std::ifstream(weight_path+"l"+std::to_string(num_layers+file_offset)+"-dense.tsv");
  ptr = 0;
  while(std::getline(MyReadFile, line)){
    output_weight[ptr++] = std::stof(line);
  }
  MyReadFile.close();
  MyReadFile = std::ifstream(bias_path+"l"+std::to_string(num_layers+file_offset)+"-dense.tsv");
  ptr = 0;
  while(std::getline(MyReadFile, line)){
    output_bias[ptr++] = std::stof(line);
  }
  MyReadFile.close();

  CHECK_ERROR(q.memcpy(_dev_output_weight, output_weight,
                   num_hidden_neurons * num_classes * sizeof(float)).wait());
  CHECK_ERROR(q.memcpy(_dev_output_bias, output_bias, num_classes * sizeof(float)).wait());

  delete [] output_weight;
  delete [] output_bias;
}

void SNICIT::_input_alloc_read(const std::string &input_path) {

  Y_input = new float[num_input*input_size];
  if (!is_cifar) {
    std::ifstream file(input_path+"t10k-images-idx3-ubyte", std::ios::binary);
    if (file.is_open())
    {
      int magic_number=0;
      int number_of_images=0;
      int n_rows=0;
      int n_cols=0;
      file.read((char*)&magic_number,sizeof(magic_number));
      file.read((char*)&number_of_images,sizeof(number_of_images));
      file.read((char*)&n_rows,sizeof(n_rows));
      file.read((char*)&n_cols,sizeof(n_cols));
      for(int i=0;i<10000;++i) // !!!!!!!!!!!!!!!!!! change!!!!!!!!!!!!!
      {
        for(int r=0;r<input_size;++r)
        {
          unsigned char temp=0;
          file.read((char*)&temp,sizeof(temp));
          Y_input[i*input_size+r]= ((float)temp)/255.0;
        }
      }
      CHECK_ERROR(_dev_Y_input = sycl::malloc_shared<float>(
                       batch_size * input_size, q));
    }
    else {
      std::cout << "ERROR: MNIST input file open failed" << std::endl;
      exit(1);
    }
    file.close();
  }
  else {
    std::ifstream file(input_path+"cifar-input.txt");
    if (file.is_open()) {
      std::string line;
      int ptr = 0;
      while(std::getline(file, line)){
        Y_input[ptr++] = std::stof(line);
      }
    }
    else {
      std::cout << "ERROR: CIFAR-10 input file open failed" << std::endl;
      exit(1);
    }
    file.close();
  }

  for(int buff=0; buff<2; buff++) {
    float *_dev_buff_Y;
    CHECK_ERROR(_dev_buff_Y = sycl::malloc_shared<float>(
                     batch_size * num_hidden_neurons, q));
    _dev_Y_hidden.emplace_back(_dev_buff_Y);
  }
  CHECK_ERROR(_dev_Y_output_whole = sycl::malloc_shared<float>(
                   num_input * num_classes, q));

  CHECK_ERROR(_dev_Y_output = sycl::malloc_shared<float>(
                   batch_size * num_classes, q));
}

void SNICIT::_result_alloc_read(const std::string &input_path) {
  int *label;
  label = new int[num_input];
  if (!is_cifar) {
    std::ifstream file(input_path+"t10k-labels-idx1-ubyte", std::ios::binary);
    if (file.is_open())
    {
      int magic_number=0;
      int number_of_images=0;
      file.read((char*)&magic_number,sizeof(magic_number));
      file.read((char*)&number_of_images,sizeof(number_of_images));
      for(int i = 0; i < 10000; ++i)
      {
        unsigned char temp=0;
        file.read((char*)&temp, sizeof(temp));
        label[i] = (int)temp;
      }
    }
    else {
      std::cout << "ERROR: MNIST result file open failed" << std::endl;
      exit(1);
    }
    file.close();
  }
  else {
    std::ifstream file;
    file.open(input_path+"cifar-label.bin", std::ios::in | std::ios::binary | std::ios::ate);
    if (!file) {
        std::cout << "ERROR: CIFAR-10 result file open failed" << std::endl;
        exit(1);
    }

    auto file_size = file.tellg();
    std::unique_ptr<char[]> buffer(new char[file_size]);

    //Read the entire file at once
    file.seekg(0, std::ios::beg);
    file.read(buffer.get(), file_size);
    file.close();

    for(std::size_t i = 0; i < num_input; ++i){
      int l = buffer[i * 3073];
      label[i] = l;
    }
  }

  CHECK_ERROR(_dev_result_label = sycl::malloc_shared<int>(num_input, q));
  CHECK_ERROR(q.memcpy(_dev_result_label, label, num_input * sizeof(int)).wait());

  delete [] label;
}

void SNICIT::_infer() {
  std::cout<<"inferring......\n";
  auto _tic = std::chrono::steady_clock::now();
  double sparse_duration = 0.0;
  double post_duration = 0.0;
  for (int round = 0; round < num_input / batch_size; round++) {
    std::cout<<"[round "<<round<<"] begins: "<<std::endl;
    if (!is_cifar) {
      CHECK_ERROR(q.memcpy(_dev_Y_input, Y_input + round * batch_size * input_size,
                       batch_size * input_size * sizeof(float)).wait());
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> local_acc(
            sycl::range<1>(sizeof(float) * num_hidden_neurons), cgh);

        const float *_dev_Y_input_ct0 = _dev_Y_input;
        const float *_dev_input_weight_ct1 = _dev_input_weight;
        const float *_dev_input_bias_ct2 = _dev_input_bias;
        auto batch_size_ct3 = batch_size;
        auto input_size_ct4 = input_size;
        auto num_hidden_neurons_ct5 = num_hidden_neurons;
        auto _dev_Y_hidden_ct6 = _dev_Y_hidden[0];

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, batch_size) *
                    sycl::range<3>(1, (int)(1024 / num_hidden_neurons),
                                   num_hidden_neurons),
                sycl::range<3>(1, (int)(1024 / num_hidden_neurons),
                               num_hidden_neurons)),
            [=](sycl::nd_item<3> item) {
              dense_input(_dev_Y_input_ct0, _dev_input_weight_ct1,
                          _dev_input_bias_ct2, batch_size_ct3, input_size_ct4,
                          num_hidden_neurons_ct5, _dev_Y_hidden_ct6, item,
                          local_acc
                              .get_multi_ptr<sycl::access::decorated::no>()
                              .get());
            });
      });
    }
    else {
      CHECK_ERROR(q.memcpy(_dev_Y_hidden[0],
                      Y_input + round * batch_size * input_size,
                      batch_size * input_size * sizeof(float)).wait());
    }

    CHECK_ERROR(q.wait());
    auto sparse_tic = std::chrono::steady_clock::now();
    // pre-convergence
    auto pre_tic = std::chrono::steady_clock::now();
    for(int cur_layer = 0; cur_layer < threshold; cur_layer++) { // num_layers-2
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> local_acc(
            sycl::range<1>(sizeof(float) * num_hidden_neurons), cgh);

        const float *_dev_Y_hidden_cur_layer_ct0 = _dev_Y_hidden[cur_layer % 2];
        const int *_dev_hidden_roffw_cur_layer_ct1 =
            _dev_hidden_roffw[cur_layer];
        const int *_dev_hidden_colsw_cur_layer_ct2 =
            _dev_hidden_colsw[cur_layer];
        const float *_dev_hidden_valsw_cur_layer_ct3 =
            _dev_hidden_valsw[cur_layer];
        const float *_dev_hidden_bias_cur_layer_ct4 =
            _dev_hidden_bias[cur_layer];
        auto batch_size_ct5 = batch_size;
        auto num_hidden_neurons_ct6 = num_hidden_neurons;
        auto num_hidden_neurons_ct7 = num_hidden_neurons;
        auto _dev_Y_hidden_cur_layer_ct8 = _dev_Y_hidden[(cur_layer + 1) % 2];

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, batch_size) *
                    sycl::range<3>(1, num_hidden_neurons,
                                   (int)(1024 / num_hidden_neurons)),
                sycl::range<3>(1, num_hidden_neurons,
                               (int)(1024 / num_hidden_neurons))),
            [=](sycl::nd_item<3> item) {
              sparse_hidden(_dev_Y_hidden_cur_layer_ct0,
                            _dev_hidden_roffw_cur_layer_ct1,
                            _dev_hidden_colsw_cur_layer_ct2,
                            _dev_hidden_valsw_cur_layer_ct3,
                            _dev_hidden_bias_cur_layer_ct4, batch_size_ct5,
                            num_hidden_neurons_ct6, num_hidden_neurons_ct7,
                            _dev_Y_hidden_cur_layer_ct8, item,
                            local_acc
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
            });
      });
      CHECK_ERROR(q.wait());

      CHECK_ERROR(q.memset(_dev_Y_hidden[cur_layer % 2], 0,
                      batch_size * num_hidden_neurons * sizeof(float)).wait());
    }
    auto pre_toc = std::chrono::steady_clock::now();
    auto pre_duration = std::chrono::duration_cast<std::chrono::microseconds>(pre_toc - pre_tic).count();
    std::cout<<"[**pre conv**] "<< pre_duration/1000.0<< "ms"<<std::endl;
    // y star generation
    auto cluster_tic = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<uint8_t, 1> local_acc(
          sycl::range<1>(sizeof(float) * (num_hidden_neurons + 2 * seed_size)), cgh);

      const float *_dev_Y_hidden_threshold_ct0 = _dev_Y_hidden[threshold % 2];
      auto y_star_row_ct1 = y_star_row;
      auto batch_size_ct2 = batch_size;
      auto num_hidden_neurons_ct3 = num_hidden_neurons;
      auto seed_size_ct4 = seed_size;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, seed_size, 1024 / seed_size),
                            sycl::range<3>(1, seed_size, 1024 / seed_size)),
          [=](sycl::nd_item<3> item) {
            y_star_gen(
                _dev_Y_hidden_threshold_ct0, y_star_row_ct1, batch_size_ct2,
                num_hidden_neurons_ct3, seed_size_ct4, item,
                local_acc.get_multi_ptr<sycl::access::decorated::no>()
                    .get());
          });
    });
    CHECK_ERROR(q.wait());

    int y_star_cnt = 0;
    for (int i = 0; i < seed_size; i++) {
      if (y_star_row[i] != -1) {
        centroid_LUT[y_star_row[i]] = -1;
        y_star_row[y_star_cnt++] = y_star_row[i];
        // printf("%drow, label=%d\n", y_star_row[i], _dev_result_label[y_star_row[i]]);
      }
    }
    // coarse cluster
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<uint8_t, 1> local_acc(
          sycl::range<1>(sizeof(float) * (num_hidden_neurons + y_star_cnt)), cgh);

      auto _dev_Y_hidden_threshold_ct0 = _dev_Y_hidden[threshold % 2];
      const int *y_star_row_ct1 = y_star_row;
      auto ne_record_ct2 = ne_record;
      auto centroid_LUT_ct4 = centroid_LUT;
      auto num_hidden_neurons_ct5 = num_hidden_neurons;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, batch_size) *
                  sycl::range<3>(1, y_star_cnt, 1024 / y_star_cnt),
              sycl::range<3>(1, y_star_cnt, 1024 / y_star_cnt)),
          [=](sycl::nd_item<3> item) {
            coarse_cluster(
                _dev_Y_hidden_threshold_ct0, y_star_row_ct1, ne_record_ct2,
                y_star_cnt, centroid_LUT_ct4, num_hidden_neurons_ct5, item,
                local_acc.get_multi_ptr<sycl::access::decorated::no>()
                    .get());
          });
    });

    CHECK_ERROR(q.wait());
    int ne_rows = 0;
    for (int i = 0; i < batch_size; i++) {
      if (ne_record[i] != false) {
        rowsY[ne_rows++] = i;
      }
    }
    // std::cout<<"non empty rows in delta Y =  "<< ne_rows <<std::endl;
    auto cluster_toc = std::chrono::steady_clock::now();
    auto cluster_duration = std::chrono::duration_cast<std::chrono::microseconds>(cluster_toc - cluster_tic).count();
    std::cout<<"[**cluster-based conversion**] finished in "<< cluster_duration/1000.0<< "ms"<<std::endl;
    // post convergence
    auto post_tic = std::chrono::steady_clock::now();
    for(int cur_layer = threshold; cur_layer < num_layers; cur_layer++) { // num_layers-2
      auto post_p_tic = std::chrono::steady_clock::now();
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> local_acc(
            sycl::range<1>(sizeof(float) * num_hidden_neurons), cgh);

        const int *rowsY_ct0 = rowsY;
        const float *_dev_Y_hidden_threshold_ct1 = _dev_Y_hidden[threshold % 2];
        const int *_dev_hidden_roffw_cur_layer_ct2 =
            _dev_hidden_roffw[cur_layer];
        const int *_dev_hidden_colsw_cur_layer_ct3 =
            _dev_hidden_colsw[cur_layer];
        const float *_dev_hidden_valsw_cur_layer_ct4 =
            _dev_hidden_valsw[cur_layer];
        auto batch_size_ct5 = batch_size;
        auto num_hidden_neurons_ct6 = num_hidden_neurons;
        auto num_hidden_neurons_ct7 = num_hidden_neurons;
        auto _dev_Y_hidden_threshold_ct8 = _dev_Y_hidden[(1 + threshold) % 2];

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, ne_rows) *
                    sycl::range<3>(1, num_hidden_neurons,
                                   (int)(1024 / num_hidden_neurons)),
                sycl::range<3>(1, num_hidden_neurons,
                               (int)(1024 / num_hidden_neurons))),
            [=](sycl::nd_item<3> item) {
              sparse_hidden_post(
                  rowsY_ct0, _dev_Y_hidden_threshold_ct1,
                  _dev_hidden_roffw_cur_layer_ct2,
                  _dev_hidden_colsw_cur_layer_ct3,
                  _dev_hidden_valsw_cur_layer_ct4, batch_size_ct5,
                  num_hidden_neurons_ct6, num_hidden_neurons_ct7,
                  _dev_Y_hidden_threshold_ct8, item,
                  local_acc
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
      CHECK_ERROR(q.wait());
      q.submit([&](sycl::handler &cgh) {
        const int *rowsY_ct0 = rowsY;
        const int *centroid_LUT_ct1 = centroid_LUT;
        const float *_dev_Y_hidden_threshold_ct2 =
            _dev_Y_hidden[(1 + threshold) % 2];
        const float *_dev_hidden_bias_cur_layer_ct3 =
            _dev_hidden_bias[cur_layer];
        auto num_hidden_neurons_ct4 = num_hidden_neurons;
        auto ne_record_ct5 = ne_record;
        auto _dev_Y_hidden_threshold_ct6 = _dev_Y_hidden[threshold % 2];

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ne_rows) *
                                  sycl::range<3>(1, 1, num_hidden_neurons),
                              sycl::range<3>(1, 1, num_hidden_neurons)),
            [=](sycl::nd_item<3> item) {
              update_post(
                  rowsY_ct0, centroid_LUT_ct1, _dev_Y_hidden_threshold_ct2,
                  _dev_hidden_bias_cur_layer_ct3, num_hidden_neurons_ct4,
                  ne_record_ct5, _dev_Y_hidden_threshold_ct6, item);
            });
      });
      CHECK_ERROR(q.wait());
      int new_ne_rows = 0;
      for (int i = 0; i < ne_rows; i++) {
        if (ne_record[rowsY[i]] != false) {
          rowsY[new_ne_rows++] = rowsY[i];
        }
      }
      ne_rows = new_ne_rows;
      // std::cout<<"non empty rows in delta Y =  "<< ne_rows <<std::endl;
      CHECK_ERROR(q.memset(_dev_Y_hidden[(1 + threshold) % 2], 0,
                       batch_size * num_hidden_neurons * sizeof(float)).wait());

      auto post_p_toc = std::chrono::steady_clock::now();
      auto post_p_duration = std::chrono::duration_cast<std::chrono::microseconds>(post_p_toc - post_p_tic).count();
      // std::cout<<"[**post convergence**]finished layer "<< cur_layer <<" in "<< post_p_duration/1000.0<< "ms"<<std::endl;
    }
    auto post_toc = std::chrono::steady_clock::now();
    post_duration = std::chrono::duration_cast<std::chrono::microseconds>(post_toc - post_tic).count();
    std::cout<<"[**post convergence**]finished in "<< post_duration/1000.0<< "ms"<<std::endl;
    // recovery
    auto recovery_tic = std::chrono::steady_clock::now();
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<uint8_t, 1> local_acc(
          sycl::range<1>(sizeof(float) * num_hidden_neurons), cgh);

      auto _dev_Y_hidden_threshold_ct0 = _dev_Y_hidden[threshold % 2];
      const int *centroid_LUT_ct1 = centroid_LUT;
      auto num_hidden_neurons_ct2 = num_hidden_neurons;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, batch_size) *
                                sycl::range<3>(1, 1, num_hidden_neurons),
                            sycl::range<3>(1, 1, num_hidden_neurons)),
          [=](sycl::nd_item<3> item) {
            recover(
                _dev_Y_hidden_threshold_ct0, centroid_LUT_ct1,
                num_hidden_neurons_ct2, item,
                local_acc.get_multi_ptr<sycl::access::decorated::no>()
                    .get());
          });
    });
    CHECK_ERROR(q.wait());
    auto recovery_toc = std::chrono::steady_clock::now();
    auto recovery_duration = std::chrono::duration_cast<std::chrono::microseconds>(recovery_toc - recovery_tic).count();
    std::cout<<"[**recovery**]finished in "<< recovery_duration/1000.0<< "ms"<<std::endl;
    auto sparse_toc = std::chrono::steady_clock::now();
    sparse_duration += std::chrono::duration_cast<std::chrono::microseconds>(sparse_toc - sparse_tic).count();

    // output layer
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<uint8_t, 1> local_acc(
          sycl::range<1>(sizeof(float) * num_classes), cgh);

      const float *_dev_Y_hidden_threshold_ct0 = _dev_Y_hidden[threshold % 2];
      const float *_dev_output_weight_ct1 = _dev_output_weight;
      const float *_dev_output_bias_ct2 = _dev_output_bias;
      auto batch_size_ct3 = batch_size;
      auto num_hidden_neurons_ct4 = num_hidden_neurons;
      auto num_classes_ct5 = num_classes;
      auto _dev_Y_output_ct6 = _dev_Y_output;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, batch_size) *
                  sycl::range<3>(1, (int)(1024 / num_classes), num_classes),
              sycl::range<3>(1, (int)(1024 / num_classes), num_classes)),
          [=](sycl::nd_item<3> item) {
            dense_output(
                _dev_Y_hidden_threshold_ct0, _dev_output_weight_ct1,
                _dev_output_bias_ct2, batch_size_ct3, num_hidden_neurons_ct4,
                num_classes_ct5, _dev_Y_output_ct6, item,
                local_acc.get_multi_ptr<sycl::access::decorated::no>()
                    .get());
          });
    });
    CHECK_ERROR(q.wait());
    CHECK_ERROR(q.memcpy(_dev_Y_output_whole + round * batch_size * num_classes,
                     _dev_Y_output, batch_size * num_classes * sizeof(float)));
  }
  // std::cout<<"SNICIT runtime: "<< sparse_duration/1000.0<< "ms"<<std::endl;
  int* cnt;
  CHECK_ERROR(cnt = sycl::malloc_shared<int>(1, q));
  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> local_acc(
        sycl::range<1>(sizeof(int)), cgh);

    auto _dev_Y_output_whole_ct0 = _dev_Y_output_whole;
    auto num_classes_ct1 = num_classes;
    auto num_input_ct2 = num_input;
    auto _dev_result_label_ct3 = _dev_result_label;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1024),
                          sycl::range<3>(1, 1, 1024)),
        [=](sycl::nd_item<3> item) {
          check_acc(
              _dev_Y_output_whole_ct0, num_classes_ct1, num_input_ct2,
              _dev_result_label_ct3, cnt, item,
              local_acc.get_multi_ptr<sycl::access::decorated::no>()
                  .get());
        });
  });
  CHECK_ERROR(q.wait());

  std::cout<<"SNICIT info: accuracy "<<100*((float)cnt[0]/(float)num_input)<<"%"<<" runtime "<< sparse_duration/1000.0<< "ms"
  <<" avgpost "<<post_duration/(1000*(num_layers-threshold))<< "ms"<<std::endl;
  auto _toc = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(_toc - _tic).count();
  std::cout<<"[Total] finished inferring in "<<duration/1000.0<< "ms"<<std::endl;
  
}

}
