# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/stevens/HeCBench/src/si-cuda/build/_deps/progressbar-src")
  file(MAKE_DIRECTORY "/home/stevens/HeCBench/src/si-cuda/build/_deps/progressbar-src")
endif()
file(MAKE_DIRECTORY
  "/home/stevens/HeCBench/src/si-cuda/build/_deps/progressbar-build"
  "/home/stevens/HeCBench/src/si-cuda/build/_deps/progressbar-subbuild/progressbar-populate-prefix"
  "/home/stevens/HeCBench/src/si-cuda/build/_deps/progressbar-subbuild/progressbar-populate-prefix/tmp"
  "/home/stevens/HeCBench/src/si-cuda/build/_deps/progressbar-subbuild/progressbar-populate-prefix/src/progressbar-populate-stamp"
  "/home/stevens/HeCBench/src/si-cuda/build/_deps/progressbar-subbuild/progressbar-populate-prefix/src"
  "/home/stevens/HeCBench/src/si-cuda/build/_deps/progressbar-subbuild/progressbar-populate-prefix/src/progressbar-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/stevens/HeCBench/src/si-cuda/build/_deps/progressbar-subbuild/progressbar-populate-prefix/src/progressbar-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/stevens/HeCBench/src/si-cuda/build/_deps/progressbar-subbuild/progressbar-populate-prefix/src/progressbar-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
