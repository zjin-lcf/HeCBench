CC = hipcc

# Specify the path to MPI
MPIPATH = /usr/lib/x86_64-linux-gnu/openmpi/

# Assume Lapack is on the system path
EXTRA_LINK_FLAGS = -fgpu-rdc --hip-link -lmpi -llapack

MPIINC = $(MPIPATH)/include
debugdir := debug_hip
optdir   := optimize_hip
single   := yes
cuda_aware_mpi := yes


CXX  = $(CC)
MPILIB = $(MPIPATH)/lib

ifeq ($(debug),yes)
   FFLAGS    = -g
   CXXFLAGS  = -g -x hip -I../src -c -DSW4_NONBLOCKING -fgpu-rdc
   CFLAGS    = -g
else
   FFLAGS   = -O3
   CXXFLAGS = -O3 -x hip -I../src -c -DSW4_NONBLOCKING -fgpu-rdc
   CFLAGS   = -O
endif
ifdef MPIINC
  CXXFLAGS += -I$(MPIINC)
endif

ifeq ($(openmp),yes)
   OMPOPT = -fopenmp
   CXXFLAGS += $(OMPOPT) -DSW4_OPENMP
   CFLAGS   += $(OMPOPT) -DSW4_OPENMP
endif

ifeq ($(cuda_aware_mpi),yes)
   CXXFLAGS += -DSW4_CUDA_AWARE_MPI
   CFLAGS   += -DSW4_CUDA_AWARE_MPI
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

OBJ  = main.o EW.o Source.o rhs4sg.o rhs4sg_rev.o SuperGrid.o GridPointSource.o time_functions_cu.o ew-cfromfort.o EW_cuda.o Sarray.o device-routines.o EWCuda.o CheckPoint.o Parallel_IO.o MaterialData.o MaterialBlock.o Polynomial.o SecondOrderSection.o TimeSeries.o sacsubc.o curvilinear-c.o rhs4sgcurv.o rhs4sgcurv_rev.o

FOBJ = $(addprefix $(builddir)/,$(OBJ))

sw4lite: $(FOBJ)
	@echo "********* User configuration variables **************"
	@echo "debug=" $(debug) " proj=" $(proj) " etree=" $(etree) " SW4ROOT"= $(SW4ROOT)
	@echo "CXX=" $(CXX) "EXTRA_CXX_FLAGS"= $(EXTRA_CXX_FLAGS)
	@echo "FC=" $(FC) " EXTRA_FORT_FLAGS=" $(EXTRA_FORT_FLAGS)
	@echo "EXTRA_LINK_FLAGS"= $(EXTRA_LINK_FLAGS)
	@echo "******************************************************"
	cd $(builddir); $(CC) -L$(MPILIB) $(OMPOPT) -Xlinker "-rpath=$(MPILIB)" -o $@ $(OBJ) $(linklibs)
	@echo "*** Build directory: " $(builddir) " ***"

$(builddir)/%.o:src/%.c
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(CXX) $(CXXFLAGS) -c ../$<

run: sw4lite
	mpirun -np 1 $(optdir)/sw4lite ../sw4lite-cuda/tests/pointsource/pointsource.in

clean:
	/bin/mkdir -p $(optdir)
	/bin/mkdir -p $(debugdir)
	cd $(optdir);
	rm -f *.o sw4lite
	cd $(debugdir)
	rm -f *.o sw4lite

