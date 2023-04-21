#ifndef cuBool_GPU_CUH
#define cuBool_GPU_CUH

#include <algorithm>
#include <vector>
#include <iostream>
#include <sstream>
#include <limits>
#include <type_traits>

//#include <omp.h>

#include "helper/rngpu.h"
#include "helper/helpers.h"

#include "config.h"  // WARPSPERBLOCK 
#include "io_and_allocation.h"
#include "bit_vector_kernels.h"

using std::ostringstream;
using std::vector;

template<typename factor_t = uint32_t>
class cuBool
{
  public:
    using factor_matrix_t = vector<factor_t>;
    using bit_vector_t = uint32_t;
    using bit_matrix_t = vector<bit_vector_t>;
    using index_t = uint32_t;
    using error_t = float;
    using cuBool_config = cuBool_config<index_t, error_t>;

  private:
    struct factor_handler {
      factor_t *d_A;
      factor_t *d_B;
      error_t *distance_;
      error_t *d_distance_;
      uint8_t factorDim_ = 20;
      size_t lineSize_ = 1;
      bool initialized_ = false;
    };

  public:
    cuBool(const bit_matrix_t& C,
           const index_t height,
           const index_t width,
           const float density,
           const size_t numActiveExperriments = 1)
    {
      std::cout << "~~~ GPU cuBool ~~~" << std::endl; 

      std::cout << "Using device : " << 
         q.get_device().get_info<sycl::info::device::name>() << std::endl;

      const int SMs = q.get_device().get_info<sycl::info::device::max_compute_units>();
      max_parallel_lines_ = SMs * WARPSPERBLOCK;

      height_ = height;
      width_ = width;

      density_ = density;
      inverse_density_ = 1 / density;

      if(std::is_same<factor_t, uint32_t>::value)
        lineSize_padded_ = 1;
      else if(std::is_same<factor_t, float>::value)
        lineSize_padded_ = 32;

      //omp_set_num_threads(numActiveExperriments);
      activeExperiments.resize(numActiveExperriments);
      bestFactors = {};

      allocate();
      std::cout << "cuBool allocation complete." << std::endl;
      std::cout << "Matrix dimensions:\t" << height_ << "x" << width_ << std::endl;

      resetBest();

      initializeMatrix(C);

      if(initialized_)
        std::cout << "cuBool initialization complete." << std::endl;
      else
        exit(1);
      std::cout << " - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << std::endl;
    }

  private:
    // allocate memory for matrix, factors and distances
    bool allocate() {
      size_t lineBytes_padded = sizeof(factor_t) * lineSize_padded_;

      for(auto& e : activeExperiments) {
        e.d_A = (factor_t*) sycl::malloc_device(lineBytes_padded * height_, q);
        e.d_B = (factor_t*) sycl::malloc_device(lineBytes_padded * width_, q);
        e.distance_ = (error_t*) sycl::malloc_host(sizeof(error_t), q);
        e.d_distance_ = (error_t*) sycl::malloc_device(sizeof(error_t), q);
      }

      bestFactors.d_A = (factor_t*)sycl::malloc_host(lineBytes_padded * height_, q);
      bestFactors.d_B = (factor_t*)sycl::malloc_host(lineBytes_padded * width_, q);
      bestFactors.distance_ = (error_t*)sycl::malloc_host(sizeof(error_t), q);

      index_t height_C = SDIV(height_, 32);
      width_C_padded_ = SDIV(width_, 32) * 32;
      d_C = (bit_vector_t*) sycl::malloc_device(sizeof(bit_vector_t) * height_C * width_C_padded_, q);

      return true;
    }

  public:
    ~cuBool() {
      sycl::free(d_C, q);
      for(auto& e : activeExperiments) {
        sycl::free(e.d_A, q);
        sycl::free(e.d_B, q);
        sycl::free(e.distance_, q);
        sycl::free(e.d_distance_, q);
      }
      sycl::free(bestFactors.d_A, q);
      sycl::free(bestFactors.d_B, q);
      sycl::free(bestFactors.distance_, q);
    }

    bool initializeMatrix(const bit_matrix_t& C) {
      if( SDIV(height_,32) * width_ != C.size()) {
        std::cerr << "cuBool construction: Matrix dimension mismatch." << std::endl;
        return false;
      }

      index_t height_C = SDIV(height_, 32);
      width_C_padded_ = SDIV(width_, 32) * 32;

      for (unsigned int i = 0; i < height_C; i++) {
        q.memcpy(d_C + i * width_C_padded_, C.data() + i * width_,
                 sizeof(bit_vector_t) * width_);
      }
      q.wait();

      return initialized_ = true;
    }

    bool resetBest() {
      *bestFactors.distance_ = std::numeric_limits<error_t>::max();
      bestFactors.initialized_ = false;
      return true;
    }

  private:
    void calculateDistance(const factor_handler &handler, const error_t weight = 1) {
      q.memset(handler.d_distance_, 0, sizeof(error_t));
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<factor_t, 1> B_block_sm(sycl::range<1>(32 * WARPSPERBLOCK), cgh);
        sycl::local_accessor<bit_vector_t, 1> C_block_sm(sycl::range<1>(32 * WARPSPERBLOCK), cgh);
        sycl::local_accessor<error_t, 1> reductionArray_sm(sycl::range<1>(WARPSPERBLOCK), cgh);
        auto d_C_t = d_C;
        auto height_t = height_;
        auto width_t = width_;
        auto width_C_padded_t = width_C_padded_;
        sycl::range<1> gws (SDIV(height_, WARPSPERBLOCK) * WARPSPERBLOCK * 32);
        sycl::range<1> lws (WARPSPERBLOCK * 32);
        cgh.parallel_for(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item)
          [[sycl::reqd_sub_group_size(32)]] {
          computeDistanceRowsShared(
            handler.d_A, handler.d_B, d_C_t, height_t,
            width_t, width_C_padded_t, handler.factorDim_,
            weight, handler.d_distance_, item,
            B_block_sm.get_pointer(),
            C_block_sm.get_pointer(),
            reductionArray_sm.get_pointer());
        });
      });
      q.memcpy(handler.distance_, handler.d_distance_, sizeof(error_t)).wait();
    }

  public:
    // initialize factors with custom Initializer function
    template <class Initializer>
    bool initializeFactors(const size_t activeId, const uint8_t factorDim, Initializer &&initilize) {
      auto& handler = activeExperiments[activeId];

      handler.factorDim_ = factorDim;

      if(std::is_same<factor_t, uint32_t>::value) {
        handler.lineSize_ = 1;
      }
      else if(std::is_same<factor_t, float>::value) {
        handler.lineSize_ = handler.factorDim_;
      }

      initilize(handler);

      *handler.distance_ = -1;

      return handler.initialized_ = true;
    }

    // initialize factors as copy of host vectors
    bool initializeFactors(const size_t activeId, const factor_matrix_t &A,
                           const factor_matrix_t &B, const uint8_t factorDim) 
    {
      return initializeFactors(activeId, factorDim, [&,this](factor_handler& handler){
        if( A.size() != height_ * handler.lineSize_ || B.size() != width_ * handler.lineSize_) {
          std::cerr << "cuBool initialization: Factor dimension mismatch." << std::endl;
          return false;
        }

        size_t lineBytes = sizeof(factor_t) * handler.lineSize_;

        // emulate 2D copy with 1D copy naively
        for (int i = 0; i < height_; i++) {
          q.memcpy(handler.d_A + i*lineSize_padded_,
                   A.data() + i*handler.lineSize_,
                   lineBytes);
        }

        for (int i = 0; i < width_; i++) {
          q.memcpy(handler.d_B + i*lineSize_padded_,
                   B.data() + i*handler.lineSize_,
                   lineBytes);
        }
      });
    }

    // initialize factors on device according to INITIALIZATIONMODE
    bool initializeFactors(const size_t activeId, const uint8_t factorDim, uint32_t seed) {
      return initializeFactors(activeId, factorDim, [&,this](factor_handler& handler) {
        float threshold = getInitChance(density_, handler.factorDim_);

        sycl::range<1> gws (SDIV(height_, WARPSPERBLOCK * 32 / lineSize_padded_) * WARPSPERBLOCK * 32);
        sycl::range<1> lws (WARPSPERBLOCK * 32);
        q.submit([&](sycl::handler &cgh) {
          auto height_t = height_;
          cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
            initFactor(handler.d_A, height_t, handler.factorDim_,
                       seed, threshold, item);
          });
        });

        seed += height_;

        sycl::range<1> gws2 (SDIV(width_, WARPSPERBLOCK * 32 / lineSize_padded_) * WARPSPERBLOCK * 32);
        q.submit([&](sycl::handler &cgh) {
          auto width_t = width_;
          cgh.parallel_for(sycl::nd_range<1>(gws2, lws), [=](sycl::nd_item<1> item) {
            initFactor(handler.d_B, width_t, handler.factorDim_,
                       seed, threshold, item);
          });
        });
      });
    }

    bool verifyDistance(const size_t activeId, const int weight = 1) {
      auto& handler = activeExperiments[activeId];

      if(!initialized_) {
        std::cerr << "cuBool matrix not initialized." << std::endl;
        return false;
      }

      error_t* distance_proof;
      error_t* d_distance_proof;

      distance_proof = (error_t*) sycl::malloc_host(sizeof(error_t), q);
      d_distance_proof = (error_t*) sycl::malloc_device(sizeof(error_t), q);
      q.memset(d_distance_proof, 0, sizeof(error_t)).wait();

      /*
              computeDistanceRowsShared<<<SDIV(height_, WARPSPERBLOCK),
                                          WARPSPERBLOCK * 32>>>(
                  handler.d_A, handler.d_B, d_C, height_, width_,
                  width_C_padded_, handler.factorDim_, weight,
                  d_distance_proof);
       */

      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<factor_t, 1> B_block_sm(sycl::range<1>(32 * WARPSPERBLOCK), cgh);
        sycl::local_accessor<bit_matrix_t, 1> C_block_sm(sycl::range<1>(32 * WARPSPERBLOCK), cgh);
        sycl::local_accessor<error_t, 1> reductionArray_sm(sycl::range<1>(WARPSPERBLOCK), cgh);

        auto d_C_t = d_C;
        auto height_t = height_;
        auto width_t = width_;
        auto width_C_padded_t = width_C_padded_;

        sycl::range<1> gws (SDIV(height_, WARPSPERBLOCK) * WARPSPERBLOCK * 32);
        sycl::range<1> lws (WARPSPERBLOCK * 32);
        cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
          computeDistanceRowsShared(
            handler.d_A, handler.d_B, d_C_t, height_t,
            width_t, width_C_padded_t, handler.factorDim_,
            weight, d_distance_proof, item,
            B_block_sm.get_pointer(),
            C_block_sm.get_pointer(),
            reductionArray_sm.get_pointer());
        });
      });
      q.memcpy(distance_proof, d_distance_proof, sizeof(error_t)).wait();

      bool equal = *handler.distance_ == *distance_proof;
      if(!equal) {
        std::cout << "----- !Distances differ! -----\n";
        std::cout << "Running distance:  " << *handler.distance_ << "\n";
        std::cout << "Real distance:     " << *distance_proof << std::endl;
      } else {
        std::cout << "Distance verified" << std::endl;
      }

      sycl::free(distance_proof, q);
      sycl::free(d_distance_proof, q);
      return equal;
    }

    void getFactors(const size_t activeId, factor_matrix_t &A, factor_matrix_t &B) const {
      auto& handler = activeExperiments[activeId];

      if(!handler.initialized_) {
        std::cerr << "Factors in slot " << activeId << " not initialized." << std::endl;
        return;
      }

      size_t lineBytes = sizeof(factor_t) * handler.lineSize_;

      A.resize(height_);

      for (int i = 0; i < height_; i++) {
        q.memcpy(A.data() + i*handler.lineSize_,
                 handler.d_A + i*lineSize_padded_,
                 lineBytes);
      }

      for (int i = 0; i < width_; i++) {
        q.memcpy(B.data() + i*handler.lineSize_,
                 handler.d_B + i*lineSize_padded_,
                 lineBytes);
      }
    }

    error_t getDistance(const size_t activeId) const {
      auto& handler = activeExperiments[activeId];

      if(!handler.initialized_) {
        std::cerr << "Factors in slot " << activeId << " not initialized." << std::endl;
        return -1;
      }
      return *handler.distance_;
    }

    // 'this' argument has type 'const sycl::queue', but method is not marked const
    void getBestFactors(factor_matrix_t &A, factor_matrix_t &B) /* const */ {
      if(!bestFactors.initialized_) {
        std::cerr << "Best result not initialized." << std::endl;
        return;
      }

      size_t lineBytes = sizeof(factor_t) * bestFactors.lineSize_;

      A.resize(height_);
      q.memcpy(A.data(), bestFactors.d_A, lineBytes * height_);

      B.resize(width_);
      q.memcpy(B.data(), bestFactors.d_B, lineBytes * width_);
    }

    error_t getBestDistance() const {
      if(!bestFactors.initialized_) {
        std::cerr << "Best result not initialized." << std::endl;
        return -1;
      }
      return *bestFactors.distance_;
    }

    void runMultiple(const size_t numExperiments, const cuBool_config& config) {
      finalDistances.resize(numExperiments);

      fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);

      #pragma omp parallel for schedule(dynamic,1) shared(state)
      for(size_t i=0; i<numExperiments; ++i) {
        //unsigned id = omp_get_thread_num();
        unsigned id = 0;
        auto config_i = config;
        uint32_t seed;
        #pragma omp critical(kiss)
        seed = fast_kiss32(state);
        #pragma omp critical(kiss)
        config_i.seed = fast_kiss32(state);
        #pragma omp critical(cout)
        std::cout << "Starting run " << i << " in slot " << id << " with seed " << config_i.seed << std::endl;
        initializeFactors(id, config_i.factorDim, seed);
        finalDistances[i] = run(id, config_i);
      }
    }

    float run(const size_t activeId, const cuBool_config &config) {
      auto& handler = activeExperiments[activeId];

      if(!initialized_) {
        std::cerr << "cuBool matrix not initialized." << std::endl;
        return -1;
      }

      if(!handler.initialized_) {
        std::cerr << "cuBool factors in slot " << activeId << " not initialized." << std::endl;
        return -1;
      }

      ostringstream out;

      calculateDistance(handler, config.weight);

      if(config.verbosity > 0) {
        out << "\tStart distance for slot " << activeId
            << "\tabs_err: " << *handler.distance_
            << "\trel_err: " << float(*handler.distance_) / height_ / width_
            << '\n';
      }

      index_t linesAtOnce = SDIV(config.linesAtOnce, WARPSPERBLOCK) * WARPSPERBLOCK;
      if(config.loadBalance) {
        linesAtOnce = linesAtOnce / max_parallel_lines_ * max_parallel_lines_;
        if (!linesAtOnce) linesAtOnce = max_parallel_lines_;
      }

      if(config.verbosity > 1) {
        out << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
        out << "- - - - Starting " << config.maxIterations
            << " GPU iterations, changing " << linesAtOnce
            << " lines each time\n";
        out << "- - - - Showing error every " << config.distanceShowEvery
          << " steps\n";
        if(config.tempStart > 0) {
          out << "- - - - Start temperature " << config.tempStart
              << " multiplied by " << config.reduceFactor
              << " every " << config.reduceStep
              << " steps\n";
          out << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
              << std::endl;
        }
      }

      fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);
      float temperature = config.tempStart;
      float weight = config.weight;
      size_t iteration = 0;
      size_t stuckIterations = 0;
      auto distancePrev = *handler.distance_;
      size_t syncStep = 100;
      while(
        *handler.distance_ > config.distanceThreshold &&
        iteration++ < config.maxIterations
        && temperature > config.tempEnd
        && stuckIterations < config.stuckIterationsBeforeBreak) {

        // Change rows
        index_t lineToBeChanged = (fast_kiss32(state) % height_) / WARPSPERBLOCK * WARPSPERBLOCK;
        uint32_t gpuSeed = fast_kiss32(state) + iteration;

        sycl::range<1> gws (SDIV(std::min(linesAtOnce, height_), WARPSPERBLOCK) * WARPSPERBLOCK * 32);
        sycl::range<1> lws (WARPSPERBLOCK * 32);

        q.submit([&](sycl::handler &cgh) {
          sycl::local_accessor<factor_t, 1> B_block_sm(sycl::range<1>(32 * WARPSPERBLOCK), cgh);
          sycl::local_accessor<bit_vector_t, 1> C_block_sm(sycl::range<1>(32 * WARPSPERBLOCK), cgh);

          auto d_C_t = d_C;
          auto height_t = height_;
          auto width_t = width_;
          auto width_C_padded_t = width_C_padded_;

          cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
            vectorMatrixMultCompareRowWarpShared
             (handler.d_A, handler.d_B, 
              d_C_t, height_t, width_t, width_C_padded_t, 
              handler.factorDim_,
              lineToBeChanged, handler.d_distance_, gpuSeed, temperature/10,
              config.flipManyChance, config.flipManyDepth, weight, item,
              B_block_sm.get_pointer(),
              C_block_sm.get_pointer());
          });
        });

        // Change cols
        lineToBeChanged = (fast_kiss32(state) % width_) / WARPSPERBLOCK * WARPSPERBLOCK;
        gpuSeed = fast_kiss32(state) + iteration;

        //vectorMatrixMultCompareColWarpShared
         // <<< SDIV(min(linesAtOnce, width_), WARPSPERBLOCK), WARPSPERBLOCK*32, 0, stream >>>
          //(handler.d_A, handler.d_B, d_C, height_, width_, width_C_padded_, handler.factorDim_,
           //lineToBeChanged, handler.d_distance_, gpuSeed, temperature/10,
           //config.flipManyChance, config.flipManyDepth, weight);

        sycl::range<1> gws2 (SDIV(std::min(linesAtOnce, width_), WARPSPERBLOCK) * WARPSPERBLOCK * 32);
        q.submit([&](sycl::handler &cgh) {

          sycl::local_accessor<factor_t, 1> A_block_sm(sycl::range<1>(32 * WARPSPERBLOCK), cgh);
          sycl::local_accessor<bit_vector_t, 1> C_block_sm(sycl::range<1>(32 * WARPSPERBLOCK), cgh);

          auto d_C_t = d_C;
          auto height_t = height_;
          auto width_t = width_;
          auto width_C_padded_t = width_C_padded_;

          cgh.parallel_for(sycl::nd_range<1>(gws2, lws), [=](sycl::nd_item<1> item) {
            vectorMatrixMultCompareColWarpShared
            (handler.d_A, handler.d_B, 
             d_C_t, height_t, width_t, width_C_padded_t, 
             handler.factorDim_,
             lineToBeChanged, handler.d_distance_, gpuSeed, temperature/10,
             config.flipManyChance, config.flipManyDepth, weight, item,
             A_block_sm.get_pointer(),
             C_block_sm.get_pointer());
           });
        });

        if(iteration % syncStep == 0) {
          q.memcpy(handler.distance_, handler.d_distance_, sizeof(error_t)).wait();

          if(*handler.distance_ == distancePrev)
            stuckIterations += syncStep;
          else
            stuckIterations = 0;
          distancePrev = *handler.distance_;
        }

        if(config.verbosity > 1 && iteration % config.distanceShowEvery == 0) {
          out << "Iteration: " << iteration
              << "\tabs_err: " << *handler.distance_
              << "\trel_err: " << float(*handler.distance_) / height_ / width_
              << "\ttemp: " << temperature;
          out << std::endl;
        }
        if(iteration % config.reduceStep == 0) {
          temperature *= config.reduceFactor;
          if(weight > 1)
            weight *= config.reduceFactor;
          if(weight < 1)
            weight = 1;
        }
      }

      if(config.verbosity > 0) {
        out << "\tBreak condition for slot " << activeId << ":\t";
        if (!(iteration < config.maxIterations))
          out << "Reached iteration limit: " << config.maxIterations;
        if (!(*handler.distance_ > config.distanceThreshold))
          out << "Distance below threshold: " << config.distanceThreshold;
        if (!(temperature > config.tempEnd))
          out << "Temperature below threshold";
        if (!(stuckIterations < config.stuckIterationsBeforeBreak))
          out << "Stuck for " << stuckIterations << " iterations";
        out << " after " << iteration << " iterations.\n";
      }

      // use hamming distance for final judgement
      calculateDistance(handler, 1);

      if(config.verbosity > 0) {
        out << "\tFinal distance for slot " << activeId
            << "\tabs_err: " << *handler.distance_
            << "\trel_err: " << float(*handler.distance_) / height_ / width_
            << std::endl;
      }

      if(*handler.distance_ < *bestFactors.distance_) {

        #pragma omp critical
        if(*handler.distance_ < *bestFactors.distance_) {
          if(config.verbosity > 0) {
            out << "\tResult is better than previous best. Copying to host." << std::endl;
          }

          *bestFactors.distance_ = *handler.distance_;
          bestFactors.lineSize_ = handler.lineSize_;
          bestFactors.factorDim_ = handler.factorDim_;

          size_t lineBytes = sizeof(factor_t) * handler.lineSize_;
 
          for (unsigned int i = 0; i < height_; i++) {
            q.memcpy(bestFactors.d_A + i*handler.lineSize_,
                     handler.d_A + i*lineSize_padded_,
                     lineBytes);
          }

          for (unsigned int i = 0; i < width_; i++) {
            q.memcpy(bestFactors.d_B + i*handler.lineSize_,
                     handler.d_B + i*lineSize_padded_,
                     lineBytes);
          }

          bestFactors.initialized_ = true;
        }
      }

      #pragma omp critical
      std::cout << out.str();

      return float(*handler.distance_) / height_ / width_;
    }

    const vector<float>& getDistances() const {
      return finalDistances;
    }

  private:
    bool initialized_ = false;
    bit_vector_t *d_C;
    float density_;
    int inverse_density_;
    index_t height_;
    index_t width_;
    index_t width_C_padded_;
    size_t lineSize_padded_;
    int max_parallel_lines_;
    factor_handler bestFactors;
    vector<factor_handler> activeExperiments;
    vector<float> finalDistances;
#ifdef USE_GPU
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
#else
    sycl::queue q{sycl::cpu_selector_v, sycl::property::queue::in_order()};
#endif

};

#endif
