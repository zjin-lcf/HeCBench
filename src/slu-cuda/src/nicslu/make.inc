CC = gcc
AR = ar -r
RANLIB = ranlib
MACROS = -DNO_ATOMIC
# Removed -DSSE2 and -msse2 for ARM64 compatibility
CFLAGS = -fPIE -c -g -O2 -Wall -Wextra $(MACROS)
LFLAGS = 
