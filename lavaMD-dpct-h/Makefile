#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = dpcpp
OPTIMIZE  = yes
DEBUG     = no
OUTPUT    = no

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp ./util/num/num.cpp ./util/timer/timer.cpp

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
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

ifeq ($(OUTPUT),yes)
  CFLAGS += -DOUTPUT
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

main.o:	main.cpp main.h kernel.h

./util/num/num.o: ./util/num/num.cpp ./util/num/num.h 

./util/timer/timer.o: ./util/timer/timer.cpp ./util/timer/timer.h 

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) -boxes1d 30

