#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = dpcpp
OPTIMIZE  = yes
DEBUG     = no

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp cOgrd.cpp  cOkadaFault.cpp  cOkadaEarthquake.cpp cSphere.cpp okada.cpp utilits.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++14 -Wall -I../include

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

cOgrd.o: cOgrd.cpp  cOgrd.h utilits.h
cOkadaEarthquake.o: cOkadaEarthquake.cpp  cOkadaEarthquake.h utilits.h
cOkadaFault.o: cOkadaFault.cpp  cOkadaFault.h utilits.h
cSphere.o: cSphere.cpp cSphere.h utilits.h
main.o : main.cpp kernels.dp.hpp easywave.h utilits.h
utilits.o: utilits.cpp utilits.h

clean:
	rm -rf $(program) $(obj) eWave.2D.*

run: $(program)
	./$(program) -grid ../easyWave-omp/data/grids/e2Asean.grd -source ../easyWave-omp/data/faults/BengkuluSept2007.flt -time 120 

