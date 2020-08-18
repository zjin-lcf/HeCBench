#ifndef common_
#define common_

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
////////////////////////////////////////////////
// Structures
////////////////////////////////////////////////

struct boundary_particle {
    sycl::double3 pos; // position
    sycl::double3 n;   // position
} ;

struct fluid_particle {
    double density;
    double pressure;
    sycl::double3 pos;    // position
    sycl::double3 v;      // velocity
    sycl::double3 v_half; // half step velocity
    sycl::double3 a;      // acceleration
};

struct param {
    double rest_density;
    double mass_particle;
    double spacing_particle;
    double smoothing_radius;
    double g;
    double time_step;
    double alpha;
    double surface_tension;
    double speed_sound;
    int number_particles;
    int number_fluid_particles;
    int number_boundary_particles;
    int number_steps;
    int steps_per_frame;
}; // Simulation paramaters

struct AABB {
    double min_x;
    double max_x;
    double min_y;
    double max_y;
    double min_z;
    double max_z;
} ; //Axis aligned bounding box



////////////////////////////////////////////////
// Function prototypes
////////////////////////////////////////////////

void constructBoundaryBox(boundary_particle *boundary_particles, AABB* boundary, param *params);
void eulerStart(fluid_particle* fluid_particles, boundary_particle *boundary_particles, param *params);
void initParticles(fluid_particle** fluid_particles, boundary_particle** boundary_particles, AABB* water, AABB* boundary, param* params);
void initParams(AABB* water_volume, AABB* boundary_volume, param* params);
void finalizeParticles(fluid_particle *fluid_particles, boundary_particle *boundary_particles);
void writeFile(fluid_particle *particles, param *params);
void writeBoundaryFile(boundary_particle *boundary, param *params);


#endif

