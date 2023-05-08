/*
 * EasyWave - A realtime tsunami simulation program with GPU support.
 * Copyright (C) 2014  Andrey Babeyko, Johannes Spazier
 * GFZ German Research Centre for Geosciences (http://www.gfz-potsdam.de)
 *
 * Parts of this program (especially the GPU extension) were developed
 * within the context of the following publicly funded project:
 * - TRIDEC, EU 7th Framework Programme, Grant Agreement 258723
 *   (http://www.tridec-online.eu)
 *
 * Licensed under the EUPL, Version 1.1 or - as soon they will be approved by
 * the European Commission - subsequent versions of the EUPL (the "Licence"),
 * complemented with the following provision: For the scientific transparency
 * and verification of results obtained and communicated to the public after
 * using a modified version of the work, You (as the recipient of the source
 * code and author of this modified version, used to produce the published
 * results in scientific communications) commit to make this modified source
 * code available in a repository that is easily and freely accessible for a
 * duration of five years after the communication of the obtained results.
 * 
 * You may not use this work except in compliance with the Licence.
 * 
 * You may obtain a copy of the Licence at:
 * https://joinup.ec.europa.eu/software/page/eupl
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the Licence is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the Licence for the specific language governing permissions and
 * limitations under the Licence.
 */

#define HEADER "\neasyWave ver.2013-04-11\n"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sycl/sycl.hpp>
#include "utilits.h"
#include "easywave.h"
#include "cOgrd.h"
#include "cOkadaEarthquake.h"

double diff(timespec start, timespec end) {

  timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }

  return (double)((double)temp.tv_nsec / 1000000000.0 + (double)temp.tv_sec);
}

int commandLineHelp( void );

int main( int argc, char **argv )
{
  char buf[1024];
  int ierr = 0;
  int argn;
  long int elapsed;
  int lastProgress,lastPropagation,lastDump;
  int loop;

  // reading parameters from a file
  FILE *fp;
  char fileLabel[5];
  unsigned short shval;
  int isBin,i,j,m,k;
  float fval;
  double dval;

  printf(HEADER);
  Err.setchannel(MSG_OUTFILE);

  // Read parameters from command line and use default
  struct EWPARAMS Par;

  // Bathymetry
  if( ( argn = utlCheckCommandLineOption( argc, argv, "grid", 4 ) ) != 0 ) {
    /* TODO: strdup not necessary here because all arguments in argv reside until program exit -> memory leak */
    Par.fileBathymetry = strdup( argv[argn+1] );
  }
  else return commandLineHelp();

  // Source: Okada faults or Surfer grid
  if( ( argn = utlCheckCommandLineOption( argc, argv, "source", 6 ) ) != 0 ) {
    Par.fileSource = strdup( argv[argn+1] );
  }
  else return commandLineHelp();

  // Simulation time, [sec]
  if( ( argn = utlCheckCommandLineOption( argc, argv, "time", 4 ) ) != 0 ) {
    Par.timeMax = atoi( argv[argn+1] );
    Par.timeMax *= 60;
  }
  else return commandLineHelp();


  // Optional parameters or their default values

  // Model name
  if( ( argn = utlCheckCommandLineOption( argc, argv, "label", 3 ) ) != 0 ) {
    Par.modelName = strdup( argv[argn+1] );
  }
  else Par.modelName = strdup( "eWave" );

  // Deactivate logging
  if( ( argn = utlCheckCommandLineOption( argc, argv, "nolog", 5 ) ) != 0 )
    ;
  else {
    Log.start( "easywave.log" );
    Log.timestamp_disable();
  }

  // Use Coriolis force
  //if( ( argn = utlCheckCommandLineOption( argc, argv, "coriolis", 3 ) ) != 0 )
  //Par.coriolis = 1;
  //else Par.coriolis = 0;

  // Periodic dumping of mariograms and cumulative 2D-plots (wavemax, arrival times), [sec]
  if( ( argn = utlCheckCommandLineOption( argc, argv, "dump", 4 ) ) != 0 )
    Par.outDump = atoi( argv[argn+1] );
  else Par.outDump = 0;

  // Reporting simulation progress, [sec model time]
  if( ( argn = utlCheckCommandLineOption( argc, argv, "progress", 4 ) ) != 0 )
    Par.outProgress = (int)(atof(argv[argn+1])*60);
  else Par.outProgress = 600;

  // 2D-wave propagation output, [sec model time]
  if( ( argn = utlCheckCommandLineOption( argc, argv, "propagation", 4 ) ) != 0 )
    Par.outPropagation = (int)(atof(argv[argn+1])*60);
  else Par.outPropagation = 300;

  // minimal calculation depth, [m]
  if( ( argn = utlCheckCommandLineOption( argc, argv, "min_depth", 9 ) ) != 0 )
    Par.dmin = (float)atof(argv[argn+1]);
  else Par.dmin = 10.;

  // timestep, [sec]
  if( ( argn = utlCheckCommandLineOption( argc, argv, "step", 4 ) ) != 0 )
    Par.dt = atoi(argv[argn+1]);
  else Par.dt = 0;  // will be estimated automatically

  // Initial uplift: relative threshold
  if( ( argn = utlCheckCommandLineOption( argc, argv, "ssh0_rel", 8 ) ) != 0 )
    Par.ssh0ThresholdRel = (float)atof(argv[argn+1]);
  else Par.ssh0ThresholdRel = 0.01;

  // Initial uplift: absolute threshold, [m]
  if( ( argn = utlCheckCommandLineOption( argc, argv, "ssh0_abs", 8 ) ) != 0 )
    Par.ssh0ThresholdAbs = (float)atof(argv[argn+1]);
  else Par.ssh0ThresholdAbs = 0.0;

  // Threshold for 2-D arrival time (0 - do not calculate), [m]
  if( ( argn = utlCheckCommandLineOption( argc, argv, "ssh_arrival", 9 ) ) != 0 )
    Par.sshArrivalThreshold = (float)atof(argv[argn+1]);
  else Par.sshArrivalThreshold = 0.001;

  // Threshold for clipping of expanding computational area, [m]
  if( ( argn = utlCheckCommandLineOption( argc, argv, "ssh_clip", 8 ) ) != 0 )
    Par.sshClipThreshold = (float)atof(argv[argn+1]);
  else Par.sshClipThreshold = 1.e-4;

  // Threshold for resetting the small ssh (keep expanding area from unnesessary growing), [m]
  if( ( argn = utlCheckCommandLineOption( argc, argv, "ssh_zero", 8 ) ) != 0 )
    Par.sshZeroThreshold = (float)atof(argv[argn+1]);
  else Par.sshZeroThreshold = 1.e-5;

  // Threshold for transparency (for png-output), [m]
  if( ( argn = utlCheckCommandLineOption( argc, argv, "ssh_transparency", 8 ) ) != 0 )
    Par.sshTransparencyThreshold = (float)atof(argv[argn+1]);
  else Par.sshTransparencyThreshold = 0.0;

  // Points Of Interest (POIs) input file
  if( ( argn = utlCheckCommandLineOption( argc, argv, "poi", 3 ) ) != 0 ) {
    Par.filePOIs = strdup( argv[argn+1] );
  }
  else Par.filePOIs = NULL;

  // POI fitting: max search distance, [km]
  if( ( argn = utlCheckCommandLineOption( argc, argv, "poi_search_dist", 15 ) ) != 0 )
    Par.poiDistMax = (float)atof(argv[argn+1]);
  else Par.poiDistMax = 10.0;
  Par.poiDistMax *= 1000.;

  // POI fitting: min depth, [m]
  if( ( argn = utlCheckCommandLineOption( argc, argv, "poi_min_depth", 13 ) ) != 0 )
    Par.poiDepthMin = (float)atof(argv[argn+1]);
  else Par.poiDepthMin = 1.0;

  // POI fitting: max depth, [m]
  if( ( argn = utlCheckCommandLineOption( argc, argv, "poi_max_depth", 13 ) ) != 0 )
    Par.poiDepthMax = (float)atof(argv[argn+1]);
  else Par.poiDepthMax = 10000.0;

  // report of POI loading
  if( ( argn = utlCheckCommandLineOption( argc, argv, "poi_report", 7 ) ) != 0 )
    Par.poiReport = 1;
  else Par.poiReport = 0;

  // POI output interval, [sec]
  if( ( argn = utlCheckCommandLineOption( argc, argv, "poi_dt_out", 10 ) ) != 0 )
    Par.poiDt = atoi(argv[argn+1]);
  else Par.poiDt = 30;

  if( ( argn = utlCheckCommandLineOption( argc, argv, "gpu", 3 ) ) != 0 )
    Par.gpu = true;
  else
    Par.gpu = false;

  if( ( argn = utlCheckCommandLineOption( argc, argv, "adjust_ztop", 11 ) ) != 0 )
    Par.adjustZtop = true;
  else
    Par.adjustZtop = false;

  if( ( argn = utlCheckCommandLineOption( argc, argv, "verbose", 7 ) ) != 0 )
    Par.verbose = true;
  else
    Par.verbose = false;


  // Log command line
  sprintf( buf, "Command line: " );
  for( argn=1; argn<argc; argn++ ) {
    strcat( buf, " " );
    strcat( buf, argv[argn] );
  }
  Log.print( "%s", buf );

  Log.print( "Loading bathymetry from %s", Par.fileBathymetry );

  // check if bathymetry file is in ascii or binary format
  if( (fp=fopen(Par.fileBathymetry,"rb")) == NULL )
    return Err.post( Err.msgOpenFile(Par.fileBathymetry) );

  memset( fileLabel, 0, 5 );
  ierr = fread( fileLabel, 4, 1, fp );
  if( !strcmp( fileLabel,"DSAA" ) )
    isBin = 0;
  else if( !strcmp( fileLabel,"DSBB" ) )
    isBin = 1;
  else
    return Err.post( "%s: not GRD-file!", Par.fileBathymetry );

  fclose(fp);

  // set the values of NLon and NLat
  int NLon, NLat;
  double LonMin, LatMin;
  double LonMax, LatMax;
  double DLon, DLat;
  double Dx, Dy;
  if( isBin ) {
    fp = fopen( Par.fileBathymetry, "rb" );
    ierr = fread( fileLabel, 4, 1, fp );
    ierr = fread( &shval, sizeof(unsigned short), 1, fp ); 
    NLon = shval;
    ierr = fread( &shval, sizeof(unsigned short), 1, fp ); 
    NLat = shval;
  } else {
    fp = fopen( Par.fileBathymetry, "rt" );
    ierr = fscanf( fp, "%s", fileLabel );
    ierr = fscanf( fp, " %d %d ", &NLon, &NLat );
  }

  // set the values of min/max Lon and Lat
  if( isBin ) {
    ierr = fread( &LonMin, sizeof(double), 1, fp ); 
    ierr = fread( &LonMax, sizeof(double), 1, fp );
    ierr = fread( &LatMin, sizeof(double), 1, fp );
    ierr = fread( &LatMax, sizeof(double), 1, fp );
    ierr = fread( &dval, sizeof(double), 1, fp );
    ierr = fread( &dval, sizeof(double), 1, fp ); // zmin zmax
  } else {
    ierr = fscanf( fp, " %lf %lf ", &LonMin, &LonMax );
    ierr = fscanf( fp, " %lf %lf ", &LatMin, &LatMax );
    ierr = fscanf( fp, " %*s %*s " );   // zmin, zmax
  }

  DLon = (LonMax - LonMin)/(NLon - 1);   // in degrees
  DLat = (LatMax - LatMin)/(NLat - 1);

  Dx = Re * g2r( DLon );     // in m along the equator
  Dy = Re * g2r( DLat );

  const size_t grid_size = (size_t)NLat*NLon*MAX_VARS_PER_NODE;
  const size_t grid_size_bytes = grid_size * sizeof(float);

  // allocate memory for GRIDNODE structure and for caching arrays
  float* node = (float*) malloc(grid_size_bytes);
  if (node == NULL) return Err.post( Err.msgAllocateMem() );
  float* R6 = (float*) malloc( sizeof(float) * (NLat+1) );
  if (R6 == NULL) return Err.post( Err.msgAllocateMem() );
  float* C1 = (float*) malloc( sizeof(float) * (NLon+1) );
  if (C1 == NULL) return Err.post( Err.msgAllocateMem() );
  float* C3 = (float*) malloc( sizeof(float) * (NLon+1) );
  if (C3 == NULL) return Err.post( Err.msgAllocateMem() );
  float* C2 = (float*) malloc( sizeof(float) * (NLat+1) );
  if (C2 == NULL) return Err.post( Err.msgAllocateMem() );
  float* C4 = (float*) malloc( sizeof(float) * (NLat+1) );
  if (C4 == NULL) return Err.post( Err.msgAllocateMem() );


  if( isBin ) {

    /* NOTE: optimal would be reading everything in one step, but that does not work because rows and columns are transposed
     * (only possible with binary data at all) - use temporary buffer for now (consumes additional memory!) */
    float *buf = new float[ NLat*NLon ];
    ierr = fread( buf, sizeof(float), NLat*NLon, fp );

    for( i=1; i<=NLon; i++ ) {
      for( j=1; j<=NLat; j++ ) {

        m = idx(j,i);

        if( isBin )
          fval = buf[ (j-1) * NLon + (i-1) ];
        //ierr = fread( &fval, sizeof(float), 1, fp );

        Node(m, iTopo) = fval;
        Node(m, iTime) = -1;
        Node(m, iD) = -fval;

        if( Node(m, iD) < 0 ) {
          Node(m, iD) = 0.0f;
        } else if( Node(m, iD) < Par.dmin ) {
          Node(m, iD) = Par.dmin;
        }

      }
    }

    delete[] buf;

  } else {

    for( j=1; j<=NLat; j++ ) {
      for( i=1; i<=NLon; i++ ) {

        m = idx(j,i);
        ierr = fscanf( fp, " %f ", &fval );

        Node(m, iTopo) = fval;
        Node(m, iTime) = -1;
        Node(m, iD) = -fval;

        if( Node(m, iD) < 0 ) {
          Node(m, iD) = 0.0f;
        } else if( Node(m, iD) < Par.dmin ) {
          Node(m, iD) = Par.dmin;
        }

      }

    }
  }

  for( k=1; k<MAX_VARS_PER_NODE-2; k++ ) {
    for( int i=1; i<=NLon; i++ ) {
      for( int j=1; j<=NLat; j++ ) {
        Node(idx(j,i), k) = 0;
      }
    }
  }

  fclose( fp );

  if( !Par.dt ) { // time step not explicitly defined

    // Make bathymetry from topography. Compute stable time step.
    double dtLoc=RealMax;

    for( i=1; i<=NLon; i++ ) {
      for( j=1; j<=NLat; j++ ) {
        m = idx(j,i);
        if( Node(m, iD) == 0.0f ) continue;
        dtLoc = My_min( dtLoc, 0.8 * (Dx*cosdeg(getLat(j))) / std::sqrt(Gravity*Node(m, iD)) );
      }
    }

    if( dtLoc > 15 ) Par.dt = 15;
    else if( dtLoc > 10 ) Par.dt = 10;
    else if( dtLoc > 5 ) Par.dt = 5;
    else if( dtLoc > 2 ) Par.dt = 2;
    else if( dtLoc > 1 ) Par.dt = 1;
    else return Err.post("Bathymetry requires too small time step (<1sec)");
    Log.print("Stable CFL time step: %g sec", dtLoc);
  }

  // Correct bathymetry for edge artefacts
  for( i=1; i<=NLon; i++ ) {
    if( Node(idx(1,i), iD) != 0 && Node(idx(2,i), iD) == 0 ) Node(idx(1,i), iD) = 0.;
    if( Node(idx(NLat,i), iD) != 0 && Node(idx(NLat-1,i), iD) == 0 ) Node(idx(NLat,i), iD) = 0.;
  }
  for( j=1; j<=NLat; j++ ) {
    if( Node(idx(j,1), iD) != 0 && Node(idx(j,2), iD) == 0 ) Node(idx(j,1), iD) = 0.;
    if( Node(idx(j,NLon), iD) != 0 && Node(idx(j,NLon-1), iD) == 0 ) Node(idx(j,NLon), iD) = 0.;
  }


  // Calculate caching grid parameters for speedup
  for( j=1; j<=NLat; j++ ) {
    R6[j] = cosdeg( LatMin + (j-0.5)*DLat );
  }

  for( i=1; i<=NLon; i++ ) {
    for( j=1; j<=NLat; j++ ) {

      m = idx(j,i);

      if( Node(m, iD) == 0 ) continue;

      Node(m, iR1) = Par.dt/Dy/R6[j];

      if( i != NLon ) {
        if( Node(m+NLat, iD) != 0 ) {
          Node(m, iR2) = 0.5*Gravity*Par.dt/Dy/R6[j]*(Node(m, iD)+Node(m+NLat, iD));
          Node(m, iR3) = 0.5*Par.dt*Omega*sindeg( LatMin + (j-0.5)*DLat );
        }
      }
      else {
        Node(m, iR2) = 0.5*Gravity*Par.dt/Dy/R6[j]*Node(m, iD)*2;
        Node(m, iR3) = 0.5*Par.dt*Omega*sindeg( LatMin + (j-0.5)*DLat );
      }

      if( j != NLat ) {
        if( Node(m+1, iD) != 0 ) {
          Node(m, iR4) = 0.5*Gravity*Par.dt/Dy*(Node(m, iD)+Node(m+1, iD));
          Node(m, iR5) = 0.5*Par.dt*Omega*sindeg( LatMin + j*DLat );
        }
      }
      /* FIXME: Bug? */
      else {
        Node(m, iR2) = 0.5*Gravity*Par.dt/Dy*Node(m, iD)*2;
        Node(m, iR3) = 0.5*Par.dt*Omega*sindeg( LatMin + j*DLat );
      }

    }
  }

  for( i=1; i<=NLon; i++ ) {
    C1[i] = 0;
    if( Node(idx(1,i), iD) != 0 ) C1[i] = 1./std::sqrt(Gravity*Node(idx(1,i), iD));
    C3[i] = 0;
    if( Node(idx(NLat,i), iD) != 0 ) C3[i] = 1./std::sqrt(Gravity*Node(idx(NLat,i), iD));
  }

  for( j=1; j<=NLat; j++ ) {
    C2[j] = 0;
    if( Node(idx(j,1), iD) != 0 ) C2[j] = 1./std::sqrt(Gravity*Node(idx(j,1), iD));
    C4[j] = 0;
    if( Node(idx(j,NLon), iD) != 0 ) C4[j] = 1./std::sqrt(Gravity*Node(idx(j,NLon), iD));
  }

  int NPOIs = 0;

  // read first record and get idea about the input type
  char record[256], /*buf[256],*/id[64];
  FILE *fpAcc,*fpRej;
  int i0,j0,imin,imax,jmin,jmax,flag,it,n;
  int rad,nmin;
  double d2,d2min,lenLon,lenLat,depth;
  double lon, lat;
  char **idPOI;
  long* idxPOI;
  int* flagRunupPOI;
  float **sshPOI;
  int* timePOI = NULL;
  int NtPOI;


  // Read points of interest
  if( Par.filePOIs != NULL ) {

    Log.print("Loading POIs from %s", Par.filePOIs );

    int MaxPOIs = utlGetNumberOfRecords( Par.filePOIs ); if( !MaxPOIs ) return Err.post( "Empty POIs file" );

    idPOI = new char*[MaxPOIs]; if( !idPOI ) return Err.post( Err.msgAllocateMem() );
    idxPOI = new long[MaxPOIs]; if( !idxPOI ) return Err.post( Err.msgAllocateMem() );
    flagRunupPOI = new int[MaxPOIs]; if( !flagRunupPOI ) return Err.post( Err.msgAllocateMem() );
    sshPOI = new float*[MaxPOIs]; if( !sshPOI ) return Err.post( Err.msgAllocateMem() );

    fp = fopen( Par.filePOIs, "rt" );
    int line = 0; 
    utlReadNextRecord( fp, record, &line );
    int itype = sscanf( record, "%s %s %s", buf, buf, buf );
    fclose( fp );

    if( itype == 2 ) { // poi-name and grid-index

      fp = fopen( Par.filePOIs, "rt" );
      line = NPOIs = 0;
      while( utlReadNextRecord( fp, record, &line ) != EOF ) {

        i = sscanf( record, "%s %d", id, &nmin );
        if( i != 2 ) {
          Log.print( "! Bad POI record: %s", record );
          continue;
        }
        idPOI[NPOIs] = strdup(id);
        idxPOI[NPOIs] = nmin;
        flagRunupPOI[NPOIs] = 1;
        NPOIs++;
      }

      fclose( fp );
      Log.print( "%d POIs of %d loaded successfully; %d POIs rejected", NPOIs, MaxPOIs, (MaxPOIs-NPOIs) );
    }
    else if( itype == 3 ) { // poi-name and coordinates

      if( Par.poiReport ) {
        fpAcc = fopen( "poi_accepted.lst", "wt" );
        fprintf( fpAcc, "ID lon lat   lonIJ latIJ depthIJ   dist[km]\n" );
        fpRej = fopen( "poi_rejected.lst", "wt" );
      }

      lenLat = My_PI*Re/180;

      fp = fopen( Par.filePOIs, "rt" );
      line = NPOIs = 0;
      while( utlReadNextRecord( fp, record, &line ) != EOF ) {

        i = sscanf( record, "%s %lf %lf %d", id, &lon, &lat, &flag );
        if( i == 3 )
          flag = 1;
        else if( i == 4 )
          ;
        else {
          Log.print( "! Bad POI record: %s", record );
          if( Par.poiReport ) fprintf( fpRej, "%s\n", record );
          continue;
        }

        // find the closest water grid node. Local distances could be 
        // treated as cartesian (2 min cell distortion at 60 degrees is only about 2 meters or 0.2%)
        i0 = (int)((lon - LonMin)/DLon) + 1;
        j0 = (int)((lat - LatMin)/DLat) + 1;
        if( i0<1 || i0>NLon || j0<1 || j0>NLat ) {
          Log.print( "!POI out of grid: %s", record );
          if( Par.poiReport ) fprintf( fpRej, "%s\n", record );
          continue;
        }
        lenLon = lenLat * R6[j0];

        for( nmin=-1,rad=0; rad<NLon && rad<NLat; rad++ ) {

          d2min = RealMax;

          imin = i0-rad; if( imin < 1 ) imin = 1;
          imax = i0+rad+1; if( imax > NLon ) imax = NLon;
          jmin = j0-rad; if( jmin < 1 ) jmin = 1;
          jmax = j0+rad+1; if( jmax > NLat ) jmax = NLat;
          for( i=imin; i<=imax; i++ )
            for( j=jmin; j<=jmax; j++ ) {
              if( i != imin && i != imax && j != jmin && j != jmax ) continue;
              n = idx(j,i);
              depth = Node(n, iD);
              if( depth < Par.poiDepthMin || depth > Par.poiDepthMax ) continue;
              d2 = std::pow( lenLon*(lon-getLon(i)), 2. ) + std::pow( lenLat*(lat-getLat(j)), 2. );
              if( d2 < d2min ) { d2min = d2; nmin = n; }
            }

          if( nmin > 0 ) break;
        }

        if( std::sqrt(d2min) > Par.poiDistMax ) {
          Log.print( "! Closest water node too far: %s", record );
          if( Par.poiReport ) fprintf( fpRej, "%s\n", record );
          continue;
        }

        idPOI[NPOIs] = strdup(id);
        idxPOI[NPOIs] = nmin;
        flagRunupPOI[NPOIs] = flag;
        NPOIs++;
        i = nmin/NLat + 1;
        j = nmin - (i-1)*NLat + 1;
        if( Par.poiReport )
          fprintf( fpAcc, "%s %.4f %.4f   %.4f %.4f %.1f   %.3f\n", 
              id, lon, lat, getLon(i), getLat(j), Node(nmin, iD), std::sqrt(d2min)/1000 );
      }

      fclose( fp );
      Log.print( "%d POIs of %d loaded successfully; %d POIs rejected", NPOIs, MaxPOIs, (MaxPOIs-NPOIs) );
      if( Par.poiReport ) {
        fclose( fpAcc );
        fclose( fpRej );
      }
    }

    // if mareograms
    if( Par.poiDt ) {
      NtPOI = Par.timeMax/Par.poiDt + 1;

      timePOI = new int[NtPOI];
      for( it=0; it<NtPOI; it++ ) timePOI[it] = -1;

      for( n=0; n<NPOIs; n++ ) {
        sshPOI[n] = new float[NtPOI];
        for( it=0; it<NtPOI; it++ ) sshPOI[n][it] = 0.;
      }
    }
  }

  // Init tsunami with faults or uplift-grid
  //ierr = ewSource(); if(ierr) return ierr;
  char dsaa_label[8];
  int srcType;
  double dz,absuzmax,absuzmin;
  cOkadaEarthquake eq;
  cOgrd uZ;

  // check input file type: GRD or fault
  if( (fp = fopen( Par.fileSource, "rb" )) == NULL ) return Err.post( Err.msgOpenFile(Par.fileSource) );
  memset( dsaa_label, 0, 5 );
  ierr = fread( dsaa_label, 4, 1, fp );
  if( !strcmp( dsaa_label,"DSAA" ) || !strcmp( dsaa_label,"DSBB" ) )
    srcType = 1;
  else
    srcType = 2;
  fclose(fp);


  // load GRD file
  if( srcType == 1) {
    ierr = uZ.readGRD( Par.fileSource ); if(ierr) return ierr;
  }

  // read fault(s) from file
  if( srcType == 2) {
    int effSymSource = 0;
    double dist,energy,factLat,effRad,effMax;

    ierr = eq.read( Par.fileSource ); if(ierr) return ierr;

    if( Par.adjustZtop ) {

      // check fault parameters
      Err.disable();
      ierr = eq.finalizeInput();
      while( ierr ) {
        i = ierr/10;
        ierr = ierr - 10*i;
        if( ierr == FLT_ERR_STRIKE ) {
          Log.print( "No strike on input: Employing effective symmetric source model" );
          if( eq.nfault > 1 ) { Err.enable(); return Err.post("Symmetric source assumes only 1 fault"); }
          eq.fault[0].strike = 0.;
          effSymSource = 1;
        }
        else if( ierr == FLT_ERR_ZTOP ) {
          Log.print( "Automatic depth correction to fault top @ 10 km" );
          eq.fault[i].depth = eq.fault[i].width/2 * sindeg(eq.fault[i].dip) + 10.e3;
        }
        else {
          Err.enable();
          return ierr;
        }
        ierr = eq.finalizeInput();
      }
      Err.enable();

    } else {

      // check fault parameters
      Err.disable();
      ierr = eq.finalizeInput();
      if( ierr ) {
        i = ierr/10;
        ierr = ierr - 10*i;
        if( ierr != FLT_ERR_STRIKE ) {
          Err.enable();
          ierr = eq.finalizeInput();
          return ierr;
        }
        Log.print( "No strike on input: Employing effective symmetric source model" );
        Err.enable();
        if( eq.nfault > 1 ) return Err.post("symmetric source assumes only 1 fault");
        eq.fault[0].strike = 0.;
        effSymSource = 1;
        ierr = eq.finalizeInput(); if(ierr) return ierr;
      }
      Err.enable();

    }

    // calculate uplift on a rectangular grid
    // set grid resolution, grid dimensions will be set automatically
    uZ.dx = DLon; uZ.dy = DLat;
    ierr = eq.calculate( uZ ); if(ierr) return ierr;

    if( effSymSource ) {
      // integrate for tsunami energy
      energy = 0.;
      for( j=0; j<uZ.ny; j++ ) {
        factLat = Dx*cosdeg(uZ.getY(0,j))*Dy;
        for( i=0; i<uZ.nx; i++ )
          energy += std::pow(uZ(i,j),2.)*factLat;
      }
      energy *= (1000*9.81/2);
      effRad = eq.fault[0].length/std::sqrt(2*M_PI);
      effMax = 1./effRad / std::sqrt(M_PI/2) / std::sqrt(1000*9.81/2) * std::sqrt(energy);
      Log.print( "Effective source radius: %g km,  max height: %g m", effRad/1000, effMax );

      // transfer uplift onto tsunami grid and define deformed area for acceleration
      for( i=0; i<uZ.nx; i++ ) {
        for( j=0; j<uZ.ny; j++ ) {
          dist = GeoDistOnSphere( uZ.getX(i,j),uZ.getY(i,j), eq.fault[0].lon,eq.fault[0].lat ) * 1000;
          if( dist < effRad ) 
            uZ(i,j) = effMax * std::cos(M_PI/2*dist/effRad);
          else
            uZ(i,j) = 0.;
        }
      }

    } // effective source

  } // src_type == fault

  // remove noise in the source
  absuzmax = uZ.getMaxAbsVal();

  if( (Par.ssh0ThresholdRel + Par.ssh0ThresholdAbs) != 0 ) {

    absuzmin = RealMax;
    if( Par.ssh0ThresholdRel != 0 ) absuzmin = Par.ssh0ThresholdRel*absuzmax;
    if( Par.ssh0ThresholdAbs != 0 && Par.ssh0ThresholdAbs < absuzmin ) absuzmin = Par.ssh0ThresholdAbs;

    for( i=0; i<uZ.nx; i++ ) {
      for( j=0; j<uZ.ny; j++ ) {
        if( std::fabs(uZ(i,j)) < absuzmin ) uZ(i,j) = 0;
      }
    }

  }

  // calculated (if needed) arrival threshold (negative value means it is relative)
  if( Par.sshArrivalThreshold < 0 ) Par.sshArrivalThreshold = absuzmax * std::fabs(Par.sshArrivalThreshold);

  // transfer uplift onto tsunami grid and define deformed area for acceleration

  // set initial min and max values
  int Imin = NLon; 
  int Imax = 1; 
  int Jmin = NLat; 
  int Jmax = 1;

  /* FIXME: change loops */
  for( i=1; i<=NLon; i++ ) {
    for( j=1; j<=NLat; j++ ) {

      lon = getLon(i);
      lat = getLat(j);

      if( Node(idx(j,i), iD) != 0. )
        dz = Node(idx(j,i), iH) = uZ.getVal( lon,lat );
      else
        dz = Node(idx(j,i), iH) = 0.;

      if( std::fabs(dz) > Par.sshClipThreshold ) {
        Imin = My_min( Imin, i );
        Imax = My_max( Imax, i );
        Jmin = My_min( Jmin, j );
        Jmax = My_max( Jmax, j );
      }

    }
  }

  if( Imin == NLon ) return Err.post( "Zero initial displacement" );

  Imin = My_max( Imin - 2, 2 );
  Imax = My_min( Imax + 2, NLon-1 );
  Jmin = My_max( Jmin - 2, 2 );
  Jmax = My_min( Jmax + 2, NLat-1 );


  Log.print( "Read source from %s", Par.fileSource );

  // Write model parameters into the log
  Log.print("\nModel parameters for this simulation:");
  Log.print("timestep: %d sec", Par.dt);
  Log.print("max time: %g min", (float)Par.timeMax/60);
  Log.print("poi_dt_out: %d sec", Par.poiDt);
  Log.print("poi_report: %s", (Par.poiReport ? "yes" : "no") );
  Log.print("poi_search_dist: %g km", Par.poiDistMax/1000.);
  Log.print("poi_min_depth: %g m", Par.poiDepthMin);
  Log.print("poi_max_depth: %g m", Par.poiDepthMax);
  //Log.print("coriolis: %s", (Par.coriolis ? "yes" : "no") );
  Log.print("min_depth: %g m", Par.dmin);
  Log.print("ssh0_rel: %g", Par.ssh0ThresholdRel);
  Log.print("ssh0_abs: %g m", Par.ssh0ThresholdAbs);
  Log.print("ssh_arrival: %g m", Par.sshArrivalThreshold);
  Log.print("ssh_clip: %g m", Par.sshClipThreshold);
  Log.print("ssh_zero: %g m", Par.sshZeroThreshold);
  Log.print("ssh_transparency: %g m\n", Par.sshTransparencyThreshold);

  int Nrec2DOutput;
  char* IndexFile;

  if( Par.outPropagation ) {
    // start index file
    sprintf( buf, "%s.2D.idx", Par.modelName );
    IndexFile = strdup(buf);

    fp = fopen( IndexFile, "wt" );

    fprintf( fp, "%g %g %d %g %g %d\n", LonMin, LonMax, NLon, LatMin, LatMax, NLat );

    fclose( fp );

    Nrec2DOutput = 0;
  }


  short nOutI;
  short nOutJ;
  double lonOutMin;
  double lonOutMax;
  double latOutMin;
  double latOutMax;
  double dtmp;
  float ftmp;

  Log.print("Starting main loop...");

  timespec start, inter, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_node = sycl::malloc_device<float>(grid_size, q);
  q.memcpy(d_node, node, grid_size_bytes);

  float *d_R6 = sycl::malloc_device<float>(NLat+1, q);
  q.memcpy(d_R6, R6, sizeof(float) * (NLat+1));

  float *d_C2 = sycl::malloc_device<float>(NLat+1, q);
  q.memcpy(d_C2, C2, sizeof(float) * (NLat+1));

  float *d_C4 = sycl::malloc_device<float>(NLat+1, q);
  q.memcpy(d_C4, C4, sizeof(float) * (NLat+1));

  float *d_C1 = sycl::malloc_device<float>(NLon+1, q);
  q.memcpy(d_C1, C1, sizeof(float) * (NLon+1));

  float *d_C3 = sycl::malloc_device<float>(NLon+1, q);
  q.memcpy(d_C3, C3, sizeof(float) * (NLon+1));

  int *d_Imin = sycl::malloc_device<int>(1, q);
  int *d_Imax = sycl::malloc_device<int>(1, q);
  int *d_Jmin = sycl::malloc_device<int>(1, q);
  int *d_Jmax = sycl::malloc_device<int>(1, q);

  for( Par.time=0,loop=1,lastProgress=Par.outProgress,lastPropagation=Par.outPropagation,lastDump=0;
      Par.time<=Par.timeMax; loop++,Par.time+=Par.dt,lastProgress+=Par.dt,lastPropagation+=Par.dt ) {

    /* FIXME: check if Par.poiDt can be used for those purposes */
    if( Par.filePOIs && Par.poiDt && ((Par.time/Par.poiDt)*Par.poiDt == Par.time) ) {
      // SavePOIs 
      q.memcpy(node, d_node, grid_size_bytes).wait();

      it = Par.time / Par.poiDt;
      timePOI[it] = Par.time;
      for( n=0; n<NPOIs; n++ ) {
        float ampFactor = 1.;
        if( flagRunupPOI[n] )
          ampFactor = std::pow( Node(idxPOI[n], iD), 0.25 );
        sshPOI[n][it] = ampFactor * Node(idxPOI[n], iH);
      }
    }

    // sea floor topography (mass conservation)
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class mass_conservation>(
        sycl::nd_range<2>(sycl::range<2>((Imax-Imin+16)/16*16, (Jmax-Jmin+16)/16*16), 
                          sycl::range<2>(16,16), sycl::id<2>(Imin, Jmin)), [=](sycl::nd_item<2> item) {
          int i = item.get_global_id(0);
          int j = item.get_global_id(1);
          if (i <= Imax && j <= Jmax) {
            int m = idx(j,i);
            if( NodeD(m, iD) == 0 ) return;
            NodeD(m, iH) = NodeD(m, iH) - NodeD(m, iR1)*( NodeD(m, iM) - NodeD(m-NLat, iM) + NodeD(m, iN)*d_R6[j] - NodeD(m-1, iN)*d_R6[j-1] );

            float absH = sycl::fabs(NodeD(m, iH));

            if( absH < Par.sshZeroThreshold ) NodeD(m, iH) = 0.;

            if( NodeD(m, iH) > NodeD(m, iHmax) ) NodeD(m, iHmax) = NodeD(m, iH);

            if( Par.sshArrivalThreshold && NodeD(m, iTime) < 0 && absH > Par.sshArrivalThreshold ) NodeD(m, iTime) = (float)Par.time;
          }
      });
    });

    q.submit([&](sycl::handler &h) {
      h.single_task<class update_bound>([=] () {
        int i, j, m;
        // open bondary conditions
        if( Jmin <= 2 ) {
          for( i=2; i<=(NLon-1); i++ ) {
            m = idx(1,i);
            NodeD(m, iH) = sycl::sqrt( sycl::pow(NodeD(m, iN), 2.0f) + 
                0.25f*sycl::pow((NodeD(m, iM)+NodeD(m-NLat, iM)),2.0f) )*d_C1[i];
            if( NodeD(m, iN) > 0 ) NodeD(m, iH) = - NodeD(m, iH);
          }
        }
        if( Imin <= 2 ) {
          for( j=2; j<=(NLat-1); j++ ) {
            m = idx(j,1);
            NodeD(m, iH) = sycl::sqrt( sycl::pow(NodeD(m, iM),2.0f) + 
                0.25f*sycl::pow((NodeD(m, iN)+NodeD(m-1, iN)),2.0f) )*d_C2[j];
            if( NodeD(m, iM) > 0 ) NodeD(m, iH) = - NodeD(m, iH);
          }
        }
        if( Jmax >= (NLat-1) ) {
          for( i=2; i<=(NLon-1); i++ ) {
            m = idx(NLat,i);
            NodeD(m, iH) = sycl::sqrt( sycl::pow(NodeD(m-1, iN),2.0f) + 0.25f*sycl::pow((NodeD(m, iM)+NodeD(m-1, iM)),2.0f) )*d_C3[i];
            if( NodeD(m-1, iN) < 0 ) NodeD(m, iH) = - NodeD(m, iH);
          }
        }
        if( Imax >= (NLon-1) ) {
          for( j=2; j<=(NLat-1); j++ ) {
            m = idx(j,NLon);
            NodeD(m, iH) = sycl::sqrt( sycl::pow(NodeD(m-NLat, iM),2.0f) + 0.25f*sycl::pow((NodeD(m, iN)+NodeD(m-1, iN)),2.0f) )*d_C4[j];
            if( NodeD(m-NLat, iM) < 0 ) NodeD(m, iH) = - NodeD(m, iH);
          }
        }
        if( Jmin <= 2 ) {
          m = idx(1,1);
          NodeD(m, iH) = sycl::sqrt( sycl::pow(NodeD(m, iM),2.0f) + sycl::pow(NodeD(m, iN),2.0f) )*d_C1[1];
          if( NodeD(m, iN) > 0 ) NodeD(m, iH) = - NodeD(m, iH);
          m = idx(1,NLon);
          NodeD(m, iH) = sycl::sqrt( sycl::pow(NodeD(m-NLat, iM),2.0f) + sycl::pow(NodeD(m, iN),2.0f) )*d_C1[NLon];
          if( NodeD(m, iN) > 0 ) NodeD(m, iH) = - NodeD(m, iH);
        }
        if( Jmin >= (NLat-1) ) {
          m = idx(NLat,1);
          NodeD(m, iH) = sycl::sqrt( sycl::pow(NodeD(m, iM),2.0f) + sycl::pow(NodeD(m-1, iN),2.0f) )*d_C3[1];
          if( NodeD(m-1, iN) < 0 ) NodeD(m, iH) = - NodeD(m, iH);
          m = idx(NLat,NLon);
          NodeD(m, iH) = sycl::sqrt( sycl::pow(NodeD(m-NLat, iM),2.0f) + sycl::pow(NodeD(m-1, iN),2.0f) )*d_C3[NLon];
          if( NodeD(m-1, iN) < 0 ) NodeD(m, iH) = - NodeD(m, iH);
        }
      });
    });

    // moment conservation
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class moment_conservation>(
      sycl::nd_range<2>(sycl::range<2>((Imax-Imin+16)/16*16, (Jmax-Jmin+16)/16*16), 
                        sycl::range<2>(16,16), sycl::id<2>(Imin, Jmin)), [=](sycl::nd_item<2> item) {
        int i = item.get_global_id(0);
        int j = item.get_global_id(1);
        if (i <= Imax && j <= Jmax) {
          int m = idx(j,i);
          if( (NodeD(m, iD)*NodeD(m+NLat, iD)) != 0 )
            NodeD(m, iM) = NodeD(m, iM) - NodeD(m, iR2)*(NodeD(m+NLat, iH)-NodeD(m, iH));

          if( (NodeD(m, iD)*NodeD(m+1, iD)) != 0 )
            NodeD(m, iN) = NodeD(m, iN) - NodeD(m, iR4)*(NodeD(m+1, iH)-NodeD(m, iH));
        }
      });
    });

    q.memcpy(d_Imin, &Imin, sizeof(int));
    q.memcpy(d_Jmin, &Jmin, sizeof(int));
    q.memcpy(d_Imax, &Imax, sizeof(int));
    q.memcpy(d_Jmax, &Jmax, sizeof(int));

    q.submit([&](sycl::handler &h) {
      h.single_task<class update_bound2>([=] () {
        // open boundaries
        int i, j, m;
        int enlarge;
        if( d_Jmin[0] <= 2 ) {
          for( i=1; i<=(NLon-1); i++ ) {
            m = idx(1,i);
            NodeD(m, iM) = NodeD(m, iM) - NodeD(m, iR2)*(NodeD(m+NLat, iH) - NodeD(m, iH));
          }
        }
        if( d_Imin[0] <= 2 ) {
          for( j=1; j<=NLat; j++ ) {
            m = idx(j,1);
            NodeD(m, iM) = NodeD(m, iM) - NodeD(m, iR2)*(NodeD(m+NLat, iH) - NodeD(m, iH));
          }
        }
        if( d_Jmax[0] >= (NLat-1) ) {
          for( i=1; i<=(NLon-1); i++ ) {
            m = idx(NLat,i);
            NodeD(m, iM) = NodeD(m, iM) - NodeD(m, iR2)*(NodeD(m+NLat, iH) - NodeD(m, iH));
          }
        }
        if( d_Imin[0] <= 2 ) {
          for( j=1; j<=(NLat-1); j++ ) {
            m = idx(j,1);
            NodeD(m, iN) = NodeD(m, iN) - NodeD(m, iR4)*(NodeD(m+1, iH) - NodeD(m, iH));
          }
        }
        if( d_Jmin[0] <= 2 ) {
          for( i=1; i<=NLon; i++ ) {
            m = idx(1,i);
            NodeD(m, iN) = NodeD(m, iN) - NodeD(m, iR4)*(NodeD(m+1, iH) - NodeD(m, iH));
          }
        }
        if( d_Imax[0] >= (NLon-1) ) {
          for( j=1; j<=(NLat-1); j++ ) {
            m = idx(j,NLon);
            NodeD(m, iN) = NodeD(m, iN) - NodeD(m, iR4)*(NodeD(m+1, iH) - NodeD(m, iH));
          }
        }

        // calculation area for the next step
        if( d_Imin[0] > 2 ) {
          for( enlarge=0, j=d_Jmin[0]; j<=d_Jmax[0]; j++ ) {
            if( sycl::fabs(NodeD(idx(j,d_Imin[0]+2), iH)) > Par.sshClipThreshold ) { enlarge = 1; break; }
          }
          if( enlarge ) { d_Imin[0]--; if( d_Imin[0] < 2 ) d_Imin[0] = 2; }
        }
        if( d_Imax[0] < (NLon-1) ) {
          for( enlarge=0, j=d_Jmin[0]; j<=d_Jmax[0]; j++ ) {
            if( sycl::fabs(NodeD(idx(j,d_Imax[0]-2), iH)) > Par.sshClipThreshold ) { enlarge = 1; break; }
          }
          if( enlarge ) { d_Imax[0]++; if( d_Imax[0] > (NLon-1) ) d_Imax[0] = NLon-1; }
        }
        if( d_Jmin[0] > 2 ) {
          for( enlarge=0, i=d_Imin[0]; i<=d_Imax[0]; i++ ) {
            if( sycl::fabs(NodeD(idx(d_Jmin[0]+2,i), iH)) > Par.sshClipThreshold ) { enlarge = 1; break; }
          }
          if( enlarge ) { d_Jmin[0]--; if( d_Jmin[0] < 2 ) d_Jmin[0] = 2; }
        }
        if( d_Jmax[0] < (NLat-1) ) {
          for( enlarge=0, i=d_Imin[0]; i<=d_Imax[0]; i++ ) {
            if( sycl::fabs(NodeD(idx(d_Jmax[0]-2,i), iH)) > Par.sshClipThreshold ) { enlarge = 1; break; }
          }
          if( enlarge ) { d_Jmax[0]++; if( d_Jmax[0] > (NLat-1) ) d_Jmax[0] = NLat-1; }
        }
      });
    });

    q.memcpy(&Imin, d_Imin, sizeof(int));
    q.memcpy(&Jmin, d_Jmin, sizeof(int));
    q.memcpy(&Imax, d_Imax, sizeof(int));
    q.memcpy(&Jmax, d_Jmax, sizeof(int));
    q.wait();

    clock_gettime(CLOCK_MONOTONIC, &inter);
    elapsed = diff(start, inter) * 1000;

    if( Par.outProgress ) {
      if( lastProgress >= Par.outProgress ) {
        printf( "Model time = %s,   elapsed: %ld msec\n", utlTimeSplitString(Par.time), elapsed );
        Log.print( "Model time = %s,   elapsed: %ld msec", utlTimeSplitString(Par.time), elapsed );
        lastProgress = 0;
      }
    }

    fflush(stdout);

    if( Par.outPropagation ) {
      if( lastPropagation >= Par.outPropagation ) {
        Nrec2DOutput++;
        fp = fopen( IndexFile, "at" );
        fprintf( fp, "%3.3d %s %d %d %d %d\n", Nrec2DOutput, utlTimeSplitString(Par.time), Imin, Imax, Jmin, Jmax );
        fclose( fp );
        lastPropagation = 0;
      }
    }
  } // main loop

  q.memcpy(node, d_node, grid_size_bytes).wait();
  sycl::free(d_node, q);
  sycl::free(d_C1, q);
  sycl::free(d_C2, q);
  sycl::free(d_C3, q);
  sycl::free(d_C4, q);
  sycl::free(d_R6, q);
  sycl::free(d_Imin, q);
  sycl::free(d_Imax, q);
  sycl::free(d_Jmin, q);
  sycl::free(d_Jmax, q);

  clock_gettime(CLOCK_MONOTONIC, &end);
  Log.print("Finishing main loop");

  // Final output
  Log.print("Final dump...");

  if (NPOIs != 0) { // Dump POIs
    if( Par.poiDt ) {  // Time series
      sprintf( buf, "%s.poi.ssh", Par.modelName );
      fp = fopen( buf, "wt" );

      fprintf( fp, "Minute" );
      for( n=0; n<NPOIs; n++ )
        fprintf( fp, "   %s", idPOI[n] );
      fprintf( fp, "\n" );

      for( it=0; (timePOI[it] != -1 && it < NtPOI); it++ ) {
        fprintf( fp, "%6.2f", (double)timePOI[it]/60 );
        for( n=0; n<NPOIs; n++ )
          fprintf( fp, " %7.3f", sshPOI[n][it] );
        fprintf( fp, "\n" );
      }

      fclose( fp );
    }

    // EAT EWH
    sprintf( buf, "%s.poi.summary", Par.modelName );
    fp = fopen( buf, "wt" );

    fprintf( fp, "ID ETA EWH\n" );

    for( n=0; n<NPOIs; n++ ) {
      fprintf( fp, "%s", idPOI[n] );
      float dbuf = Node(idxPOI[n], iTime)/60;
      if( dbuf < 0. ) dbuf = -1.;
      fprintf( fp, " %6.2f", dbuf );

      float ampFactor = 1.;
      if( flagRunupPOI[n] )
        ampFactor = pow( Node(idxPOI[n], iD), 0.25 );

      fprintf( fp, " %6.3f\n", (ampFactor * Node(idxPOI[n], iHmax)) );
    }
    fclose( fp );
  }

  //ewDump2D();
  nOutI = Imax-Imin+1;
  lonOutMin = getLon(Imin); lonOutMax = getLon(Imax);
  nOutJ = Jmax-Jmin+1;
  latOutMin = getLat(Jmin); latOutMax = getLat(Jmax);

  // write ssh max
  sprintf( record, "%s.2D.sshmax", Par.modelName );
  fp = fopen( record, "wb" );
  fwrite( "DSBB", 4, 1, fp );
  fwrite( &nOutI, sizeof(short), 1, fp );
  fwrite( &nOutJ, sizeof(short), 1, fp );
  fwrite( &lonOutMin, sizeof(double), 1, fp );
  fwrite( &lonOutMax, sizeof(double), 1, fp );
  fwrite( &latOutMin, sizeof(double), 1, fp );
  fwrite( &latOutMax, sizeof(double), 1, fp );
  dtmp = 0.; fwrite( &dtmp, sizeof(double), 1, fp );
  dtmp = 1.; fwrite( &dtmp, sizeof(double), 1, fp );
  for( j=Jmin; j<=Jmax; j++ ) {
    for( i=Imin; i<=Imax; i++ ) {
      ftmp = (float)Node(idx(j,i), iHmax);
      fwrite( &ftmp, sizeof(float), 1, fp );
    }
  }
  fclose( fp );

  // write arrival times
  sprintf( record, "%s.2D.time", Par.modelName );
  fp = fopen( record, "wb" );
  fwrite( "DSBB", 4, 1, fp );
  fwrite( &nOutI, sizeof(short), 1, fp );
  fwrite( &nOutJ, sizeof(short), 1, fp );
  fwrite( &lonOutMin, sizeof(double), 1, fp );
  fwrite( &lonOutMax, sizeof(double), 1, fp );
  fwrite( &latOutMin, sizeof(double), 1, fp );
  fwrite( &latOutMax, sizeof(double), 1, fp );
  dtmp = 0.; fwrite( &dtmp, sizeof(double), 1, fp );
  dtmp = 1.; fwrite( &dtmp, sizeof(double), 1, fp );
  for( j=Jmin; j<=Jmax; j++ ) {
    for( i=Imin; i<=Imax; i++ ) {
      ftmp = (float)Node(idx(j,i), iTime) / 60;  // -1/60
      //printf("%f\n", ftmp);
      fwrite( &ftmp, sizeof(float), 1, fp );
    }
  }
  fclose( fp );

  free( node );
  free( R6 );
  free( C1 );
  free( C2 );
  free( C3 );
  free( C4 );

  printf_v("Runtime: %.3lf\n", diff(start, end) * 1000.0);
  return 0;
}


//========================================================================
int commandLineHelp( void )
{
  printf( "Usage: easywave  -grid ...  -source ...  -time ... [optional parameters]\n" );
  printf( "-grid ...         bathymetry in GoldenSoftware(C) GRD format (text or binary)\n" );
  printf( "-source ...       input wave either als GRD-file or file with Okada faults\n" );
  printf( "-time ...         simulation time in [min]\n" );
  printf( "Optional parameters:\n" );
  printf( "-step ...         simulation time step, default- estimated from bathymetry\n" );
  //printf( "-coriolis         use Coriolis force, default- no\n" );
  printf( "-poi ...          POIs file\n" );
  printf( "-label ...        model name, default- 'eWave'\n" );
  printf( "-progress ...     show simulation progress each ... minutes, default- 10\n" );
  printf( "-propagation ...  write wave propagation grid each ... minutes, default- 5\n" );
  printf( "-dump ...         make solution dump each ... physical seconds, default- 0\n" );
  printf( "-nolog            deactivate logging\n" );
  printf( "-poi_dt_out ...   output time step for mariograms in [sec], default- 30\n" );
  printf( "-poi_search_dist ...  in [km], default- 10\n" );
  printf( "-poi_min_depth ...    in [m], default- 1\n" );
  printf( "-poi_max_depth ...    in [m], default- 10 000\n" );
  printf( "-poi_report       enable POIs loading report, default- disabled\n" );
  printf( "-ssh0_rel ...     relative threshold for initial wave, default- 0.01\n" );
  printf( "-ssh0_abs ...     absolute threshold for initial wave in [m], default- 0\n" );
  printf( "-ssh_arrival ...  threshold for arrival times in [m], default- 0.001\n" );
  printf( "                  negative value considered as relative threshold\n" );
  printf( "-gpu              start GPU version of EasyWave (requires a CUDA capable device)\n" );
  printf( "-verbose          generate verbose output on stdout\n" );
  printf( "\nExample:\n" );
  printf( "\t easyWave -grid gebcoIndonesia.grd  -source fault.inp  -time 120\n\n" );

  return -1;
}
