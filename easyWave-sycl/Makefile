include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp cOgrd.cpp  cOkadaFault.cpp  cOkadaEarthquake.cpp \
         cSphere.cpp okada.cpp utilits.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

cOgrd.o: cOgrd.cpp  cOgrd.h utilits.h
cOkadaEarthquake.o: cOkadaEarthquake.cpp  cOkadaEarthquake.h utilits.h
cOkadaFault.o: cOkadaFault.cpp  cOkadaFault.h utilits.h
cSphere.o: cSphere.cpp cSphere.h utilits.h
main.o : main.cpp easywave.h utilits.h
utilits.o: utilits.cpp utilits.h

clean:
	rm -rf $(program) $(obj) eWave.2D.*

run: $(program)
	./$(program) -grid ../easyWave-omp/data/grids/e2Asean.grd \
                     -source ../easyWave-omp/data/faults/BengkuluSept2007.flt \
                     -time 120 

