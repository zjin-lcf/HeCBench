#include <sycl/sycl.hpp>
#include "seed.hpp"

int
getAlignScore(SeedL const &myseed){
  return myseed.score;
}

int
getBeginPositionH(SeedL const &myseed){
  return myseed.beginPositionH;
}

int
getBeginPositionV(SeedL const &myseed){
  return myseed.beginPositionV;
}

int
getEndPositionH(SeedL const &myseed){
  return myseed.endPositionH;
}

int
getEndPositionV(SeedL const &myseed){
  return myseed.endPositionV;
}

int
getSeedLLength(SeedL const &myseed){
  return myseed.seedLength;
}

SYCL_EXTERNAL int getLowerDiagonal(SeedL const &myseed) {
  return myseed.lowerDiagonal;
}

SYCL_EXTERNAL int getUpperDiagonal(SeedL const &myseed) {
  return myseed.upperDiagonal;
}

int
getBeginDiagonal(SeedL const &myseed){
  return myseed.beginDiagonal;
}

int
getEndDiagonal(SeedL const &myseed){
  return myseed.endDiagonal;
}

void
setAlignScore(SeedL &myseed,int const value){
  myseed.score = value;
}

void
setBeginPositionH(SeedL &myseed,int const value){
  myseed.beginPositionH = value;
}

void
setBeginPositionV(SeedL &myseed,int const value){
  myseed.beginPositionV = value;
}

void
setEndPositionH(SeedL &myseed,int const value){
  myseed.endPositionH = value;
}

void
setEndPositionV(SeedL &myseed,int const value){
  myseed.endPositionV = value;
}

void
setSeedLLength(SeedL &myseed,int const value){
  myseed.seedLength = value;
}

SYCL_EXTERNAL void setLowerDiagonal(SeedL &myseed, int const value) {
  myseed.lowerDiagonal = value;
}

SYCL_EXTERNAL void setUpperDiagonal(SeedL &myseed, int const value) {
  myseed.upperDiagonal = value;
}

void
setBeginDiagonal(SeedL &myseed,int const value){
  myseed.beginDiagonal = value;
}

void
setEndDiagonal(SeedL &myseed,int const value){
  myseed.endDiagonal = value;
}
