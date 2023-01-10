//==================================================================
// Title:  C++ x-drop seed-and-extend alignment algorithm
// Author: A. Zeni, G. Guidi
//==================================================================
#ifndef __SEED_CUH__
#define __SEED_CUH__

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cassert>

template<typename Tx_>
const Tx_&  min_logan(const Tx_& _Left, const Tx_& Right_)
{   // return smaller of _Left and Right_
  if (_Left < Right_)
    return _Left;
  else
    return Right_;
}

template<typename Tx_, typename Ty_>
Tx_  min_logan(const Tx_& _Left, const Ty_& Right_)
{   // return smaller of _Left and Right_
  return (Right_ < _Left ? Right_ : _Left);
}

template<typename Ty_> 
Ty_ const &
max_logan(const Ty_& _Left, const Ty_& Right_)
{   // return larger of _Left and Right_
  if (_Left < Right_)
    return Right_;
  else
    return _Left;
}

template<typename Tx_, typename Ty_>
Tx_
max_logan(const Tx_& _Left, const Ty_& Right_)
{   // return smaller of _Left and Right_
  return (Right_ < _Left ? _Left : Right_);
}


struct SeedL
{
  int beginPositionH;
  int beginPositionV;
  int endPositionH;
  int endPositionV;
  int seedLength;
  int lowerDiagonal;  
  int upperDiagonal;  
  int beginDiagonal;
  int endDiagonal;
  int score;

  SeedL(): beginPositionH(0), beginPositionV(0), endPositionH(0), endPositionV(0), lowerDiagonal(0), upperDiagonal(0), score(0)
  {}

  SeedL(int beginPositionH, int beginPositionV, int seedLength):
    beginPositionH(beginPositionH), beginPositionV(beginPositionV), endPositionH(beginPositionH + seedLength),
    endPositionV(beginPositionV + seedLength), lowerDiagonal((beginPositionH - beginPositionV)),
    upperDiagonal((beginPositionH - beginPositionV)), beginDiagonal(beginPositionH - beginPositionV),
    endDiagonal(endPositionH - endPositionV), score(0)
  {
    assert(upperDiagonal >= lowerDiagonal);
  }

  SeedL(int beginPositionH, int beginPositionV, int endPositionH, int endPositionV):
    beginPositionH(beginPositionH),
    beginPositionV(beginPositionV),
    endPositionH(endPositionH),
    endPositionV(endPositionV),
    lowerDiagonal(min_logan((beginPositionH - beginPositionV), (endPositionH - endPositionV))),
    upperDiagonal(max_logan((beginPositionH - beginPositionV), (endPositionH - endPositionV))),
    beginDiagonal((beginPositionH - beginPositionV)),
    endDiagonal((endPositionH - endPositionV)),
    score(0)
  {
    assert(upperDiagonal >= lowerDiagonal);
  }

  SeedL(SeedL const& other):
    beginPositionH(other.beginPositionH),
    beginPositionV(other.beginPositionV),
    endPositionH(other.endPositionH),
    endPositionV(other.endPositionV),
    lowerDiagonal(other.lowerDiagonal),
    upperDiagonal(other.upperDiagonal),
    beginDiagonal(other.beginDiagonal),
    endDiagonal(other.endDiagonal),
    score(0)
  {
    assert(upperDiagonal >= lowerDiagonal);
  }

};

struct Result
{
  SeedL myseed;
  int score;       // alignment score
  int length;      // overlap length / max extension

  Result() : score(0), length(0)//check
  {
  }

  Result(int kmerLen) : score(0), length(kmerLen)
  {
  }

};

int getAlignScore(SeedL const &myseed);

int getBeginPositionH(SeedL const &myseed);

int getBeginPositionV(SeedL const &myseed);

int getEndPositionH(SeedL const &myseed);

int getEndPositionV(SeedL const &myseed);

int getSeedLLength(SeedL const &myseed);

SYCL_EXTERNAL int getLowerDiagonal(SeedL const &myseed);

SYCL_EXTERNAL int getUpperDiagonal(SeedL const &myseed);

int getBeginDiagonal(SeedL const &myseed);

int getEndDiagonal(SeedL const &myseed);

void setAlignScore(SeedL &myseed,int const value);

void setBeginPositionH(SeedL &myseed,int const value);

void setBeginPositionV(SeedL &myseed,int const value);

void setEndPositionH(SeedL &myseed,int const value);

void setEndPositionV(SeedL &myseed,int const value);

void setSeedLLength(SeedL &myseed,int const value);

SYCL_EXTERNAL void setLowerDiagonal(SeedL &myseed, int const value);

SYCL_EXTERNAL void setUpperDiagonal(SeedL &myseed, int const value);

void setBeginDiagonal(SeedL &myseed,int const value);

void setEndDiagonal(SeedL &myseed,int const value);

#endif
