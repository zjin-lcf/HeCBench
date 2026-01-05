# halo-finder-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME halo-finder
    MODEL sycl
    SOURCES ForceTreeTest.cxx
            ParticleDistribute.cxx
            ParticleExchange.cxx
            InitialExchange.cxx            
            Message.cxx
            ForceLaw.cxx
            BHForceTree.cxx
            RCOForceTree.cxx
            RCBForceTree.cxx
            Partition.cxx
            Timings.cxx
            Timer.cxx
            bigchunk.c
            dims-local.c
            BGQStep16.c
            BGQCM.c
    CATEGORIES algorithms
    INCLUDE_DIRS dfft
    COMPILE_OPTIONS -DRCB_UNTHREADED_BUILD -DUSE_SERIAL_COSMO -DID_64 -DPOSVEL_32 -DGRID_32 -DLONG_INTEGER
    LINK_LIBRARIES mpi
)
