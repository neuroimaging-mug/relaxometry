/* 
 * File:   Complex.cuh
 * Author: christiantinauer
 *
 * Created on March 5, 2015, 1:47 PM
 */

#ifndef COMPLEX_CUH
#define	COMPLEX_CUH

#include <cuComplex.h>

typedef  cuComplex c;

__device__ inline c operator +(c a, c b)
{
  return cuCaddf(a, b);
}

 __device__ inline c operator -(c a, c b)
{
  return cuCsubf(a, b);
}

__device__ inline c operator *(c a, c b)
{
  return cuCmulf(a, b);
}

__device__ inline c operator /(c a, c b)
{
  return cuCdivf(a, b);
}

__device__ inline  float abs(c a)
{
  return cuCabsf(a);
}

// Implementation from c++ 4.4.5 complex line 850
__device__ inline  c sqrt(c a)
{
	c result;

  float x = a.x;
  float y = a.y;

  if (x == 0.0f)
  {
    float t = sqrt(abs(y) / 2);
    result.x = t;
    result.y = y < 0.0f ? -t : t;
  }
  else
  {
    float t = sqrt(2 * (abs(a) + abs(x)));
    float u = t / 2;
    if (x > 0.0f)
    {
      result.x = u;
      result.y = y / t;
    }
    else
    {
      result.x = abs(y) / t;
      result.y = y < 0.0f ? -u : u;
    }
  }

  return result;
}

#endif	/* COMPLEX_CUH */