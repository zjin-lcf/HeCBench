#include "StructuredEikonal.h"

StructuredEikonal::StructuredEikonal(sycl::queue &q, bool verbose) 
  :verbose_(verbose), isGpuMemCreated_(false),
  width_(256), height_(256), depth_(256),
  itersPerBlock_(10), solverType_(0), q(q) {}

StructuredEikonal::~StructuredEikonal() {}

void StructuredEikonal::writeNRRD(std::string filename) {
  std::fstream out(filename.c_str(), std::ios::out | std::ios::binary);
  out << "NRRD0001\n";
  out << "# Complete NRRD file format specification at:\n";
  out << "# http://teem.sourceforge.net/nrrd/format.html\n";
  out << "type: double\n";
  out << "dimension: 3\n";
  out << "sizes: " << this->width_ << " " << this->height_ << " " << this->depth_ << "\n";
  out << "endian: little\n";
  out << "encoding: raw\n\n";
  double checksum = 0.0;
  for(size_t k = 0; k < this->depth_; k++) {
    for(size_t j = 0; j < this->height_; j++) {
      for(size_t i = 0; i < this->width_; i++) {
        double d = this->answer_[i][j][k];
        checksum += d;
        out.write(reinterpret_cast<const char*>(&d),sizeof(double));
      }
    }
  }
  out.close();
  printf("Checksum = %lf\n", checksum / (this->depth_ * this->height_ * this->width_));
}

void StructuredEikonal::setDims(size_t x, size_t y, size_t z) {
  this->width_ = x;
  this->height_ = y;
  this->depth_ = z;
}

void StructuredEikonal::error(char* msg) {
  printf("%s\n",msg);
  assert(false);
  exit(0);
}

void StructuredEikonal::init_device_mem() {
  assert(this->width_ > 0 && this->height_ > 0 && this->depth_ > 0);
  if(this->width_ <= 0 || this->height_ <= 0 || this->depth_ <= 0){ 
    printf("Volume dimension cannot be zero");
    exit(1);
  }

  // 1. Create /initialize GPU memory
  size_t nx, ny, nz;

  nx = this->width_ + (BLOCK_LENGTH-this->width_%BLOCK_LENGTH)%BLOCK_LENGTH;
  ny = this->height_ + (BLOCK_LENGTH-this->height_%BLOCK_LENGTH)%BLOCK_LENGTH;
  nz = this->depth_ + (BLOCK_LENGTH-this->depth_%BLOCK_LENGTH)%BLOCK_LENGTH;
  if (this->verbose_) {
    printf("%zu %zu %zu\n",nx,ny,nz);
  }

  auto volSize = nx*ny*nz;
  auto blkSize = BLOCK_LENGTH*BLOCK_LENGTH*BLOCK_LENGTH;

  auto nBlkX = nx / BLOCK_LENGTH;
  auto nBlkY = ny / BLOCK_LENGTH;
  auto nBlkZ = nz / BLOCK_LENGTH;
  auto blockNum = nBlkX*nBlkY*nBlkZ;

  this->memoryStruct_.xdim = static_cast<int>(nx);
  this->memoryStruct_.ydim = static_cast<int>(ny);
  this->memoryStruct_.zdim = static_cast<int>(nz);
  this->memoryStruct_.volsize = static_cast<uint>(volSize);
  this->memoryStruct_.blksize = static_cast<uint>(blkSize);
  this->memoryStruct_.blklength = BLOCK_LENGTH;
  this->memoryStruct_.blknum = static_cast<uint>(blockNum);
  this->memoryStruct_.nIter = static_cast<int>(this->itersPerBlock_); // iter per block

  if(this->isGpuMemCreated_) // delete previous memory
  {
    free((DOUBLE*)this->memoryStruct_.h_sol);
    free((uint*)this->memoryStruct_.h_list);
    free((bool*)this->memoryStruct_.h_listed);
    free((bool*)this->memoryStruct_.h_listVol);
    free((int*)this->memoryStruct_.blockOrder);
    sycl::free(this->memoryStruct_.d_spd, q);
    sycl::free(this->memoryStruct_.d_sol, q);
    sycl::free(this->memoryStruct_.t_sol, q);  // temp solution for ping-pong
    sycl::free(this->memoryStruct_.d_con, q);  // convergence volume
    sycl::free(this->memoryStruct_.d_list, q);
    sycl::free(this->memoryStruct_.d_listVol, q);
    sycl::free(this->memoryStruct_.d_mask, q);
  }
  this->isGpuMemCreated_ = true;

  this->memoryStruct_.h_sol = (DOUBLE*) malloc(volSize*sizeof(DOUBLE)); // initial solution
  this->memoryStruct_.h_list = (uint*) malloc(blockNum*sizeof(uint)); // linear list contains active block indices
  this->memoryStruct_.h_listed = (bool*) malloc(blockNum*sizeof(bool));  // whether block is added to the list
  this->memoryStruct_.h_listVol = (bool*) malloc(blockNum*sizeof(bool)); // volume list shows active/nonactive of corresponding block
  this->memoryStruct_.blockOrder = (int*) malloc(blockNum*sizeof(int));

  //
  // create host/device memory
  //
  this->memoryStruct_.d_spd = sycl::malloc_device<double>(volSize, q);

  this->memoryStruct_.d_sol = sycl::malloc_device<DOUBLE>(volSize, q);

  this->memoryStruct_.t_sol = sycl::malloc_device<DOUBLE>(volSize, q);  // temp solution for ping-pong

  this->memoryStruct_.d_con = sycl::malloc_device<bool>(volSize, q);  // convergence volume

  this->memoryStruct_.d_list = sycl::malloc_device<uint>(blockNum, q);

  this->memoryStruct_.d_listVol = sycl::malloc_device<bool>(blockNum, q);

  this->memoryStruct_.d_mask = sycl::malloc_device<bool>(volSize, q);
}

void StructuredEikonal::set_attribute_mask() {
  uint volSize = this->memoryStruct_.volsize;

  int nx, ny, nz, blklength;

  nx = memoryStruct_.xdim;
  ny = memoryStruct_.ydim;
  nz = memoryStruct_.zdim;
  blklength = memoryStruct_.blklength;

  // create host memory
  double *h_spd  = new double[volSize]; // byte speed, host
  bool  *h_mask = new bool[volSize];

  // copy input volume to host memory
  // make each block to be stored contiguously in 1D memory space
  uint idx = 0;
  for(int zStr = 0; zStr < nz; zStr += blklength) {
    for(int yStr = 0; yStr < ny; yStr += blklength) {
      for(int xStr = 0; xStr < nx; xStr += blklength) {
        // for each block
        for(int z=zStr; z<zStr+blklength; z++) {
          for(int y=yStr; y<yStr+blklength; y++) {
            for(int x=xStr; x<xStr+blklength; x++) {
              h_spd[idx] = this->speeds_[x][y][z];
              h_mask[idx] = true;
              idx++;
            }
          }
        }
      }
    }
  }

  // initialize GPU memory with host memory
  q.memcpy(memoryStruct_.d_spd, h_spd, volSize*sizeof(double));
  q.memcpy(memoryStruct_.d_mask, h_mask, volSize*sizeof(bool));
  q.wait();

  delete[] h_spd;
  delete[] h_mask;
}

void StructuredEikonal::initialization() {
  this->init_device_mem();
  this->set_attribute_mask();
}

void StructuredEikonal::map_generator() {
  double pi = 3.141592653589793238462643383;
  this->speeds_ = std::vector<std::vector<std::vector<double> > >(
    this->width_, std::vector<std::vector<double> >(
    this->height_, std::vector<double>(this->depth_,1.)));
  switch(this->solverType_){
  case 0 :
    //Constant Speed Map
    break;
  case 1 :
    //Sinusoid Speed Map
    for (size_t k = 0 ; k < this->depth_ ; ++k) {
      for (size_t j = 0 ; j < this->height_; ++j) {
        for ( size_t i = 0 ; i < this->width_ ; ++i) {
          this->speeds_[i][j][k] =
            (6 + 5*(sin((i*pi)/this->width_ *2))*
            sin((j*pi)/this->height_*2)*
            sin((k*pi)/this->depth_*2));
        }
      }
    }
    break;
  }
}

void StructuredEikonal::setSeeds(std::vector<std::array<size_t, 3> > seeds) {
  this->seeds_ = seeds;
}

void StructuredEikonal::useSeeds() {
  if (this->verbose_) {
    std::cout << "Loading seed volume..." << std::endl;
  }
  uint volSize, blockNum;
  int nx, ny, nz, blklength;

  nx = this->memoryStruct_.xdim;
  ny = this->memoryStruct_.ydim;
  nz = this->memoryStruct_.zdim;
  volSize = this->memoryStruct_.volsize;
  blklength = this->memoryStruct_.blklength;
  blockNum = this->memoryStruct_.blknum;

  // copy input volume to host memory
  // make each block to be stored contiguously in 1D memory space
  uint idx = 0;
  uint blk_idx = 0;
  uint list_idx = 0;
  uint nActiveBlock = 0;

  for(int zStr = 0; zStr < nz; zStr += blklength) {
    for(int yStr = 0; yStr < ny; yStr += blklength) {
      for(int xStr = 0; xStr < nx; xStr += blklength) {
        // for each block
        bool isSeedBlock = false;

        for(int z=zStr; z<zStr+blklength; z++) {
          for(int y=yStr; y<yStr+blklength; y++) {
            for(int x=xStr; x<xStr+blklength; x++) {
              this->memoryStruct_.h_sol[idx] = INF;
              if (this->seeds_.empty()) {
                if (x == nx/2 && y == ny/2 && z == nz/2) {
                  this->memoryStruct_.h_sol[idx] = 0;
                  isSeedBlock = true;
                  if (this->verbose_) {
                    printf("%d is Selected bt source \n",idx);
                  }
                }
              } else {
                for(size_t i = 0; i < this->seeds_.size(); i++) {
                  if (this->seeds_[i][0] == (size_t)x && 
                      this->seeds_[i][1] == (size_t)y && 
                      this->seeds_[i][2] == (size_t)z) {
                    this->memoryStruct_.h_sol[idx] = 0;
                    isSeedBlock = true;
                    if (this->verbose_) {
                      printf("%d is Selected bt source \n",idx);
                    }
                  }
                }
              }
              idx++;
            }
          }
        }
        ///////////////////////////////////////////////
        if(isSeedBlock) {
          if (this->verbose_) {
            printf("%d,%d,%d is Seed Block \n",zStr,yStr,xStr);
          }
          this->memoryStruct_.h_listVol[blk_idx] = true;
          this->memoryStruct_.h_listed[blk_idx] = true;
          this->memoryStruct_.h_list[list_idx] = blk_idx;
          list_idx++;
          nActiveBlock++;
        } else {
          this->memoryStruct_.h_listVol[blk_idx] = false;
          this->memoryStruct_.h_listed[blk_idx] = false;
        }
        blk_idx++;
      }
    }
  }
  this->memoryStruct_.nActiveBlock = nActiveBlock;
  // initialize GPU memory with host memory
  q.memcpy(this->memoryStruct_.d_sol, this->memoryStruct_.h_sol, volSize*sizeof(DOUBLE));
  q.memcpy(this->memoryStruct_.t_sol, this->memoryStruct_.h_sol, volSize*sizeof(DOUBLE));
  q.memcpy(this->memoryStruct_.d_list, this->memoryStruct_.h_list, nActiveBlock*sizeof(uint));
  q.memcpy(this->memoryStruct_.d_listVol, this->memoryStruct_.h_listVol, blockNum*sizeof(bool));
  // initialize GPU memory with constant value
  q.memset(this->memoryStruct_.d_con, 1, volSize*sizeof(bool));
}

void StructuredEikonal::setMapType(size_t t) {
    this->solverType_ = t;
  }

void StructuredEikonal::solveEikonal() {
  if (this->speeds_.empty()) {
    this->map_generator();
  }
  this->isGpuMemCreated_ = false;
  this->initialization();
  this->useSeeds();
  runEikonalSolverSimple(q, this->memoryStruct_);
  this->get_solution();
}

std::vector< std::vector< std::vector<double> > > 
  StructuredEikonal::getFinalResult() {
    return this->answer_;
  }

void StructuredEikonal::get_solution() {
  // copy solution from GPU
  q.memcpy(this->memoryStruct_.h_sol,
    this->memoryStruct_.d_sol, this->memoryStruct_.volsize*sizeof(DOUBLE)).wait();

  //put the data where it belongs in the grand scheme of data!
  this->answer_ = std::vector<std::vector<std::vector<double> > >(
    this->width_, std::vector<std::vector<double> >( 
    this->height_, std::vector<double>(this->depth_,0)));
  for(size_t blockID = 0; blockID < this->memoryStruct_.blknum; blockID++) {
    size_t baseAddr = blockID * this->memoryStruct_.blksize;
		size_t xgridlength = this->memoryStruct_.xdim/BLOCK_LENGTH;
		size_t ygridlength = this->memoryStruct_.ydim/BLOCK_LENGTH;
		// compute block index
		size_t bx = blockID%xgridlength;
		size_t tmpIdx = (blockID - bx)/xgridlength;
		size_t by = tmpIdx%ygridlength;
		size_t bz = (tmpIdx-by)/ygridlength;
    //translate back to real space
    for(int k = 0; k < BLOCK_LENGTH; k++) {
      for(int j = 0; j < BLOCK_LENGTH; j++) {
        for(int i = 0; i < BLOCK_LENGTH; i++) {
          double d = this->memoryStruct_.h_sol[baseAddr + 
            k * BLOCK_LENGTH * BLOCK_LENGTH + 
            j * BLOCK_LENGTH + i];
          if ((i + bx * BLOCK_LENGTH) < this->width_ && 
            (j + by * BLOCK_LENGTH) < this->height_ &&
            (k + bz * BLOCK_LENGTH) < this->depth_) {
            this->answer_[(i + bx * BLOCK_LENGTH)][(j +
              by * BLOCK_LENGTH)][k + bz * BLOCK_LENGTH] = d;
          }
        }
      }
    }
  }
}

void StructuredEikonal::setItersPerBlock(size_t t) {
  this->itersPerBlock_ = t;
}
