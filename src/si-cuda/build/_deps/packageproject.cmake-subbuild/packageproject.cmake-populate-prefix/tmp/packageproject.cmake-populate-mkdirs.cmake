# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/stevens/HeCBench/src/si-cuda/build/_deps/packageproject.cmake-src")
  file(MAKE_DIRECTORY "/home/stevens/HeCBench/src/si-cuda/build/_deps/packageproject.cmake-src")
endif()
file(MAKE_DIRECTORY
  "/home/stevens/HeCBench/src/si-cuda/build/_deps/packageproject.cmake-build"
  "/home/stevens/HeCBench/src/si-cuda/build/_deps/packageproject.cmake-subbuild/packageproject.cmake-populate-prefix"
  "/home/stevens/HeCBench/src/si-cuda/build/_deps/packageproject.cmake-subbuild/packageproject.cmake-populate-prefix/tmp"
  "/home/stevens/HeCBench/src/si-cuda/build/_deps/packageproject.cmake-subbuild/packageproject.cmake-populate-prefix/src/packageproject.cmake-populate-stamp"
  "/home/stevens/HeCBench/src/si-cuda/build/_deps/packageproject.cmake-subbuild/packageproject.cmake-populate-prefix/src"
  "/home/stevens/HeCBench/src/si-cuda/build/_deps/packageproject.cmake-subbuild/packageproject.cmake-populate-prefix/src/packageproject.cmake-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/stevens/HeCBench/src/si-cuda/build/_deps/packageproject.cmake-subbuild/packageproject.cmake-populate-prefix/src/packageproject.cmake-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/stevens/HeCBench/src/si-cuda/build/_deps/packageproject.cmake-subbuild/packageproject.cmake-populate-prefix/src/packageproject.cmake-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
