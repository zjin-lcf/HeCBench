#    /* 
#  * -- High Performance Computing Linpack Benchmark (HPL)                
#  *    Modifications Copyright (C) 2023 Intel Corporation​
#  *                                                                      
#  * -- Copyright notice and Licensing terms:                             
#  *                                                                      
#  * Redistribution  and  use in  source and binary forms, with or without
#  * modification, are  permitted provided  that the following  conditions
#  * are met:                                                             
#  *                                                                      
#  * 1. Redistributions  of  source  code  must retain the above copyright
#  * notice, this list of conditions and the following disclaimer.        
#  *                                                                      
#  * 2. Redistributions in binary form must reproduce  the above copyright
#  * notice, this list of conditions,  and the following disclaimer in the
#  * documentation and/or other materials provided with the distribution. 
#  *                                                                      
#  * 3. All  advertising  materials  mentioning  features  or  use of this
#  * software must display the following acknowledgement:                 
#  * This  product  includes  software  developed  at  the  University  of
#  * Tennessee, Knoxville, Innovative Computing Laboratory.             
#  *                                                                      
#  * 4. The name of the  University,  the name of the  Laboratory,  or the
#  * names  of  its  contributors  may  not  be used to endorse or promote
#  * products  derived   from   this  software  without  specific  written
#  * permission.                                                          
#  *                                                                      
#  * -- Disclaimer:                                                       
#  *                                                                      
#  * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
#  * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
#  * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
#  * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
#  * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
#  * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
#  * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
#  * ---------------------------------------------------------------------
#  */ 

# /* 
#  * -- High Performance Computing Linpack Benchmark (HPL)                
#  *    HPL - 2.3 - December 2, 2018                          
#  *    Antoine P. Petitet                                                
#  *    University of Tennessee, Knoxville                                
#  *    Innovative Computing Laboratory                                 
#  *    (C) Copyright 2000-2008 All Rights Reserved                       
#  *                                                                      
#  * -- Copyright notice and Licensing terms:                             
#  *                                                                      
#  * Redistribution  and  use in  source and binary forms, with or without
#  * modification, are  permitted provided  that the following  conditions
#  * are met:                                                             
#  *                                                                      
#  * 1. Redistributions  of  source  code  must retain the above copyright
#  * notice, this list of conditions and the following disclaimer.        
#  *                                                                      
#  * 2. Redistributions in binary form must reproduce  the above copyright
#  * notice, this list of conditions,  and the following disclaimer in the
#  * documentation and/or other materials provided with the distribution. 
#  *                                                                      
#  * 3. All  advertising  materials  mentioning  features  or  use of this
#  * software must display the following acknowledgement:                 
#  * This  product  includes  software  developed  at  the  University  of
#  * Tennessee, Knoxville, Innovative Computing Laboratory.             
#  *                                                                      
#  * 4. The name of the  University,  the name of the  Laboratory,  or the
#  * names  of  its  contributors  may  not  be used to endorse or promote
#  * products  derived   from   this  software  without  specific  written
#  * permission.                                                          
#  *                                                                      
#  * -- Disclaimer:                                                       
#  *                                                                      
#  * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
#  * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
#  * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
#  * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
#  * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
#  * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
#  * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
#  * ---------------------------------------------------------------------
#  */ 


all: libdgemm.so.1.0.1

OBJS = cuda_dgemm.o 

.PRECIOUS: $(OBJS)

all : libdgemm.so.1.0.1 

fermi_dgemm.o : fermi_dgemm.c fermi_dgemm.h 

DEFINES = -DMPI -g
#DEFINES += -DUSE_FERMI_DGEMM
#DEFINES += -DVERBOSE_PRINT
#DEFINES += -DACML
#DEFINES += -DGOTO

%.o: %.cpp
	mpicc -O3 -c -fPIC $(DEFINES) $*.cpp -o $*.o -I/opt/rocm/include -D__HIP_PLATFORM_AMD__

libdgemm.so.1.0.1: $(OBJS)

	mpicc -O3 -shared -Wl,-soname,libdgemm.so.1 -o libdgemm.so.1.0.1 $(OBJS) -L/opt/rocm/lib -lhipblas
	ln -sf libdgemm.so.1.0.1 libdgemm.so.1.0
	ln -sf libdgemm.so.1.0 libdgemm.so.1
	ln -sf libdgemm.so.1 libdgemm.so

clean:
	rm -f $(OBJS) $(CUBINS) libdgemm.so.1.0.1 libdgemm.so.1.0 libdgemm.so.1 libdgemm.so
