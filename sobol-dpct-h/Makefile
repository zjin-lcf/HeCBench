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

program = SobolQRNG

source = sobol.dp.cpp sobol_gold.dp.cpp sobol_gpu.dp.cpp sobol_primitives.dp.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++14 -Wall

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

$(program): $(obj) 
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

sobol.dp.o:sobol.dp.cpp
	$(CC) $(CFLAGS) -c $< -o $@

sobol_gold.dp.o:sobol_gold.dp.cpp
	$(CC) $(CFLAGS) -c $< -o $@

sobol_gpu.dp.o:sobol_gpu.dp.cpp
	$(CC) $(CFLAGS) -c $< -o $@

sobol_primitives.dp.o:sobol_primitives.dp.cpp
	$(CC) $(CFLAGS) -c $< -o $@


run: $(program)
	./$(program) 1000000 1000

clean:
	rm -rf $(program) $(obj)

