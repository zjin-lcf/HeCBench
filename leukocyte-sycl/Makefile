include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================
# Matrix library path
MATRIX_DIR = ./meschach_lib

#===============================================================================
# Targets to Build
#===============================================================================

leukocyte: detect_main.o \
	avilib.o \
	track_ellipse.o \
	track_ellipse_sycl.o \
	find_ellipse.o \
	misc_math.o \
	helper.o \
	$(MATRIX_DIR)/meschach.a
	$(CC) $(CFLAGS) -o leukocyte avilib.o \
		track_ellipse.o \
		track_ellipse_sycl.o \
		find_ellipse.o \
		misc_math.o \
		detect_main.o \
		helper.o \
		$(MATRIX_DIR)/meschach.a $(LDFLAGS)

%.o: %.[c]
	$(CC) $(OUTPUT) $(CFLAGS) $< -c

detect_main.o: detect_main.c \
               kernel_GICOV.sycl kernel_dilated.sycl \
               track_ellipse.h helper.h misc_math.h common.h

find_ellipse.o: find_ellipse.c find_ellipse.h avilib.h misc_math.h

track_ellipse.o: track_ellipse.c find_ellipse.h $(MATRIX_DIR)/matrix.h avilib.h helper.h

track_ellipse_sycl.o: track_ellipse_sycl.c track_ellipse.h common.h kernel_IMGVF.sycl


# Use the sycl compiler to build the library
# Cleanup everything, then clean everything except the static library
$(MATRIX_DIR)/meschach.a:
	cd $(MATRIX_DIR); make cleanup; make all CC=$(CC) CFLAGS=-O3; make clean

clean:
	rm -f *.o leukocyte 
	cd $(MATRIX_DIR); make cleanup

run: leukocyte
	./leukocyte ../data/leukocyte/testfile.avi 100

