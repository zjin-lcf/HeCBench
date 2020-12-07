//  SW4 LICENSE
// # ----------------------------------------------------------------------
// # SW4 - Seismic Waves, 4th order
// # ----------------------------------------------------------------------
// # Copyright (c) 2013, Lawrence Livermore National Security, LLC. 
// # Produced at the Lawrence Livermore National Laboratory. 
// # 
// # Written by:
// # N. Anders Petersson (petersson1@llnl.gov)
// # Bjorn Sjogreen      (sjogreen2@llnl.gov)
// # 
// # LLNL-CODE-643337 
// # 
// # All rights reserved. 
// # 
// # This file is part of SW4, Version: 1.0
// # 
// # Please also read LICENCE.txt, which contains "Our Notice and GNU General Public License"
// # 
// # This program is free software; you can redistribute it and/or modify
// # it under the terms of the GNU General Public License (as published by
// # the Free Software Foundation) version 2, dated June 1991. 
// # 
// # This program is distributed in the hope that it will be useful, but
// # WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
// # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
// # conditions of the GNU General Public License for more details. 
// # 
// # You should have received a copy of the GNU General Public License
// # along with this program; if not, write to the Free Software
// # Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA 
#include "EWCuda.h"
#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
using namespace std;

//-----------------------------------------------------------------------
EWCuda::EWCuda( int ndevice, int nstream )
{
   m_ndevice = ndevice;
   m_nstream = nstream;
   m_active_gpu = 0;
   if( nstream > 0 )
      m_stream  = new hipStream_t[nstream];
   else
      m_stream = static_cast<hipStream_t*>(0);
}

//-----------------------------------------------------------------------
void EWCuda::reset_gpu()
{
   if( m_ndevice > 0 )
      hipDeviceReset();
}

//-----------------------------------------------------------------------
void EWCuda::sync_stream( int st )
{
   hipError_t retcode;
   retcode = hipStreamSynchronize(m_stream[st]);
   if( retcode != hipSuccess )
      cout << "Error EWCuda::EWCuda, hipStreamSynchronize no " << st << " returned " <<
	 hipGetErrorString(retcode) << endl;
}

int mystrcmp(void const *a, void const *b) { 
  char const *aa = (char const *)a;
  char const *bb = (char const *)b;

  return strcmp(aa, bb);
}

//-----------------------------------------------------------------------
void EWCuda::sync_device( )
{
   hipError_t retcode;
   retcode = hipDeviceSynchronize();
   if( retcode != hipSuccess )
      cout << "Error EWCuda::EWCuda, hipDeviceSynchronize returned " <<
	 hipGetErrorString(retcode) << endl;
}

extern "C"
void setupgpu(int verbose)
{
  int rank, nrank;
  MPI_Comm_size(MPI_COMM_WORLD, &nrank);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  char (*hosts)[MPI_MAX_PROCESSOR_NAME] = (char (*)[MPI_MAX_PROCESSOR_NAME])malloc(nrank*(sizeof *hosts));
  char hostname[MPI_MAX_PROCESSOR_NAME];
  int namelen;
  
  // each rank collects hostname of all nodes
  MPI_Get_processor_name(hostname, &namelen);
  strcpy(hosts[rank], hostname);
  for (int i = 0; i < nrank; i++) {
    MPI_Bcast(hosts[i], MPI_MAX_PROCESSOR_NAME, MPI_CHAR, i, MPI_COMM_WORLD);
  }

  // sort list of names
  qsort(hosts, nrank, MPI_MAX_PROCESSOR_NAME, mystrcmp);

  // assign same color to the same node
  int color = 0;
  for (int i = 0; i < nrank; i++) {
    if (i > 0) {
      if (strcmp(hosts[i-1], hosts[i]) != 0) color++;
    }
    if (strcmp(hosts[i], hostname) == 0) break;
  }
  
  MPI_Comm new_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, 0, &new_comm);
  int new_rank;
  MPI_Comm_rank(new_comm, &new_rank);

  int ngpu;
  hipError_t ierr;
  ierr = hipGetDeviceCount(&ngpu);
  if (ierr != hipSuccess) {
    printf("%s\n", hipGetErrorString(ierr));
    exit(0);
  }

  int igpu = new_rank % ngpu;
  if (verbose) 
    printf("P(%d): %s using gpu id %d\n", rank, hosts[rank], igpu);
  ierr = hipSetDevice(igpu);
  if (ierr != hipSuccess) {
    printf("%s\n", hipGetErrorString(ierr));
    exit(0);
  }

  free(hosts);
}

//-----------------------------------------------------------------------
void EWCuda::initialize_gpu(int myrank)
{
   if( m_ndevice > 0){
     setupgpu(0);
     hipError_t retcode;
     for (int s = 0; s < m_nstream; s++)
     {
       retcode = hipStreamCreate(&m_stream[s]);
       if( retcode != hipSuccess )
         cout << "Error EWCuda::EWCuda, hipStreamCreate no " << s << " returned " <<
           hipGetErrorString(retcode) << endl;
     }
     
   }
}

//-----------------------------------------------------------------------
EWCuda::~EWCuda()
{
   for( int s = 0 ; s < m_nstream ; s++ )
      hipStreamDestroy(m_stream[s]);
   if( m_nstream > 0 )
      delete[] m_stream;
}
