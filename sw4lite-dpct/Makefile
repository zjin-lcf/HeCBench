CC = dpcpp
MPIPATH = $(dir $(shell which mpicc))../
EXTRA_LINK_FLAGS = -lmpi -llapack
MPIINC = $(MPIPATH)/include
debugdir := debug_sycl
optdir   := optimize_sycl
sycl_aware_mpi := yes
single := yes
CXX  = $(CC)
MPILIB = $(MPIPATH)/lib

ifeq ($(debug),yes)
   FFLAGS    = -g
   CXXFLAGS  = -g -I../src -c -DSW4_NONBLOCKING
   CFLAGS    = -g
else
   FFLAGS   = -O3
   CXXFLAGS = -O3 -I../src -c -DSW4_NONBLOCKING
   CFLAGS   = -O
endif
ifdef MPIINC
  CXXFLAGS += -I$(MPIINC)
endif

#FFLAGS   += -Msycl

ifeq ($(openmp),yes)
   OMPOPT = -fopenmp
   CXXFLAGS += $(OMPOPT) -DSW4_OPENMP
   CFLAGS   += $(OMPOPT) -DSW4_OPENMP
endif

ifeq ($(sycl_aware_mpi),yes)
   CXXFLAGS +=  -DSW4_CUDA_AWARE_MPI
   CFLAGS   +=  -DSW4_CUDA_AWARE_MPI
endif

ifeq ($(openmp),yes)
   debugdir := $(debugdir)_mp
   optdir   := $(optdir)_mp
endif
ifeq ($(single),yes)
   debugdir := $(debugdir)_sp
   optdir := $(optdir)_sp
   CXXFLAGS += -I../src/float
else
   CXXFLAGS += -I../src/double
endif

ifdef EXTRA_CXX_FLAGS
   CXXFLAGS += $(EXTRA_CXX_FLAGS)
endif

ifdef EXTRA_FORT_FLAGS
   FFLAGS += $(EXTRA_FORT_FLAGS)
endif

ifdef EXTRA_LINK_FLAGS
   linklibs += $(EXTRA_LINK_FLAGS)
endif

ifeq ($(debug),yes)
   builddir = $(debugdir)
else
   builddir = $(optdir)
endif

OBJ  = main.o \
       EW.C.dp.o \
       Source.C.dp.o \
       rhs4sg.o \
       rhs4sg_rev.o \
       SuperGrid.C.dp.o \
       GridPointSource.C.dp.o \
       time_functions_cu.C.dp.o \
       ew-cfromfort.C.dp.o \
       EW_cuda.C.dp.o \
       Sarray.C.dp.o \
       device-routines.C.dp.o \
       EWCuda.C.dp.o \
       CheckPoint.C.dp.o \
       Parallel_IO.o \
       MaterialData.o \
       MaterialBlock.o \
       Polynomial.o \
       SecondOrderSection.o \
       TimeSeries.C.dp.o \
       sacsubc.o \
       curvilinear-c.o \
       rhs4sgcurv.o \
       rhs4sgcurv_rev.o


FOBJ = $(addprefix $(builddir)/,$(OBJ))

sw4lite: $(FOBJ)
	@echo "********* User configuration variables **************"
	@echo "debug=" $(debug) " proj=" $(proj) " etree=" $(etree) " SW4ROOT"= $(SW4ROOT)
	@echo "CXX=" $(CXX) "EXTRA_CXX_FLAGS"= $(EXTRA_CXX_FLAGS)
	@echo "FC=" $(FC) " EXTRA_FORT_FLAGS=" $(EXTRA_FORT_FLAGS)
	@echo "EXTRA_LINK_FLAGS"= $(EXTRA_LINK_FLAGS)
	@echo "******************************************************"
	cd $(builddir); $(CC) -L$(MPILIB) $(OMPOPT) -o $@ $(OBJ) $(linklibs) 
	@echo "*** Build directory: " $(builddir) " ***"

$(builddir)/%.o:src/%.cpp
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(CXX) $(CXXFLAGS) -c ../$<

$(builddir)/%.o:src/%.C
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(CXX) $(CXXFLAGS) -c ../$<


clean: 
	/bin/mkdir -p $(optdir)
	/bin/mkdir -p $(debugdir)
	cd $(optdir);/bin/rm -f *.o sw4lite
	cd $(debugdir);/bin/rm -f *.o sw4lite

