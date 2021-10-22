# include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "lebesgue.h"

// Type 1 Chebyshev points.
double *chebyshev1 ( int n )
{
  double angle;
  int i;
  const double r8_pi = 3.141592653589793;
  double *x;

  x = ( double * ) malloc ( n * sizeof ( double ) );

  for ( i = 0; i < n; i++ )
  {
    angle = r8_pi * ( double ) ( 2 * i + 1 ) / ( double ) ( 2 * n );
    x[i] = cos ( angle );
  }
  return x;
}

// Type 2 Chebyshev points.
double *chebyshev2 ( int n )
{
  double angle;
  int i;
  const double r8_pi = 3.141592653589793;
  double *x;

  x = ( double * ) malloc ( n * sizeof ( double ) );

  if ( n == 1 )
  {
    x[0] = 0.0;
  }
  else
  {
    for ( i = 0; i < n; i++ )
    {
      angle = r8_pi * ( double ) ( n - i - 1 ) / ( double ) ( n - 1 );
      x[i] = cos ( angle );
    }
  }

  return x;
}

// the Type 3 Chebyshev points.
double *chebyshev3 ( int n )
{
  double angle;
  int i;
  const double r8_pi = 3.141592653589793;
  double *x;

  x = ( double * ) malloc ( n * sizeof ( double ) );

  for ( i = 0; i < n; i++ )
  {
    angle = r8_pi * ( double ) ( 2 * n - 2 * i - 1 ) 
      / ( double ) ( 2 * n         + 1 );
    x[i] = cos ( angle );
  }

  return x;
}

// the Type 4 Chebyshev points.
double *chebyshev4 ( int n )
{
  double angle;
  int i;
  const double r8_pi = 3.141592653589793;
  double *x;

  x = ( double * ) malloc ( n * sizeof ( double ) );

  for ( i = 0; i < n; i++ )
  {
    angle = r8_pi * ( double ) ( 2 * n - 2 * i )
      / ( double ) ( 2 * n + 1 );
    x[i] = cos ( angle );
  }

  return x;
}

// Type 1 Equidistant points.
double *equidistant1 ( int n )
{
  int i;
  double *x;

  x = ( double * ) malloc ( n * sizeof ( double ) );

  for ( i = 0; i < n; i++ )
  {
    x[i] = ( double ) ( - n + 1 + 2 * i ) / ( double ) ( n + 1 );
  }

  return x;
}

// Type 2 Equidistant points.
double *equidistant2 ( int n )
{
  int i;
  double *x;

  x = ( double * ) malloc ( n * sizeof ( double ) );

  if ( n == 1 )
  {
    x[0] = 0.0;
  }
  else
  {
    for ( i = 0; i < n; i++ )
    {
      x[i] = ( double ) ( - n + 1 + 2 * i ) / ( double ) ( n - 1 );
    }
  }

  return x;
}

// Type 3 Equidistant points.
double *equidistant3 ( int n )
{
  int i;
  double *x;

  x = ( double * ) malloc ( n * sizeof ( double ) );

  for ( i = 0; i < n; i++ )
  {
    x[i] = ( double ) ( - n + 1 + 2 * i ) / ( double ) ( n );
  }

  return x;
}

// the Type 1 Fejer points.
double *fejer1 ( int n )
{
  int i;
  const double r8_pi = 3.141592653589793;
  double theta;
  double *x;

  x = ( double * ) malloc ( n * sizeof ( double ) );

  for ( i = 0; i < n; i++ )
  {
    theta = r8_pi * ( double ) ( 2 * n - 1 - 2 * i ) 
      / ( double ) ( 2 * n );
    x[i] = cos ( theta );
  }
  return x;
}

// the Type 2 Fejer points.
double *fejer2 ( int n )
{
  int i;
  const double r8_pi = 3.141592653589793;
  double theta;
  double *x;

  x = ( double * ) malloc ( n * sizeof ( double ) );

  for ( i = 0; i < n; i++ )
  {
    theta = r8_pi * ( double ) ( n - i ) 
      / ( double ) ( n + 1 );
    x[i] = cos ( theta );
  }

  return x;
}


/* estimates the Lebesgue constant for a set of points.
  Parameters:
  Input, the number of interpolation points.
  Input, interpolation points.
  Input, the number of evaluation points.
  Input, evaluation points.
  Output, an estimate of the Lebesgue constant for the points.
*/
double lebesgue_constant ( int n, double x[], int nfun, double xfun[] )

{
  double *lfun;
  double lmax;

  lfun = lebesgue_function ( n, x, nfun, xfun );

  lmax = r8vec_max ( nfun, lfun );

  free ( lfun );

  return lmax;
}
   

/* create a vector of linearly spaced values.

    An R8VEC is a vector of R8's.

    4 points evenly spaced between 0 and 12 will yield 0, 4, 8, 12.
 
    In other words, the interval is divided into N-1 even subintervals,
    and the endpoints of intervals are used as the points.

  Parameters:

    Input, the number of entries in the vector.

    Input, first and last entries.

    Output, a vector of linearly spaced data.
*/
double *r8vec_linspace_new ( int n, double a, double b )

{
  int i;
  double *x;

  x = ( double * ) malloc ( n * sizeof ( double ) );

  if ( n == 1 )
  {
    x[0] = ( a + b ) / 2.0;
  }
  else
  {
    for ( i = 0; i < n; i++ )
    {
      x[i] = ( ( double ) ( n - 1 - i ) * a 
             + ( double ) (         i ) * b ) 
             / ( double ) ( n - 1     );
    }
  }
  return x;
}

// the value of the maximum element in a R8VEC.
double r8vec_max ( int n, double r8vec[] )
{
  int i;
  double value;

  if ( n <= 0 )
  {
    value = 0.0;
    return value;
  }

  value = r8vec[0];

  for ( i = 1; i < n; i++ )
  {
    if ( value < r8vec[i] )
    {
      value = r8vec[i];
    }
  }
  return value;
}

void r8vec_print ( int n, double a[], char *title )
{
  int i;

  fprintf ( stdout, "\n" );
  fprintf ( stdout, "%s\n", title );
  fprintf ( stdout, "\n" );
  for ( i = 0; i < n; i++ )
  {
    fprintf ( stdout, "  %8d: %14g\n", i, a[i] );
  }

  return;
}

void timestamp ( )
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  fprintf ( stdout, "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}

