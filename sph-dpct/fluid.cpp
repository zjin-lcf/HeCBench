#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "common.h"

////////////////////////////////////////////////////////////////////////////
// B spline smoothing kernel
////////////////////////////////////////////////////////////////////////////

double W(sycl::double3 p_pos, sycl::double3 q_pos, double h)
{
    double r = sycl::sqrt(
        (static_cast<double>(p_pos.x()) - static_cast<double>(q_pos.x())) *
            (static_cast<double>(p_pos.x()) - static_cast<double>(q_pos.x())) +
        (static_cast<double>(p_pos.y()) - static_cast<double>(q_pos.y())) *
            (static_cast<double>(p_pos.y()) - static_cast<double>(q_pos.y())) +
        (static_cast<double>(p_pos.z()) - static_cast<double>(q_pos.z())) *
            (static_cast<double>(p_pos.z()) - static_cast<double>(q_pos.z())));
    double C = 1.0/(M_PI*h*h*h);
    double u = r/h;
    double val = 0.0;
    if(u >= 2.0)
        return val;
    else if(u < 1.0 )
        val = 1.0 - (3.0/2.0)*u*u + (3.0/4.0)*u*u*u;
    else if(u >= 1.0 && u < 2.0)
        val = (1.0 / 4.0) * sycl::pow(2.0 - u, 3.0);

    val *= C;
    return val;
}

// Gradient of B spline kernel

double del_W(sycl::double3 p_pos, sycl::double3 q_pos, double h)
{
    double r = sycl::sqrt(
        (static_cast<double>(p_pos.x()) - static_cast<double>(q_pos.x())) *
            (static_cast<double>(p_pos.x()) - static_cast<double>(q_pos.x())) +
        (static_cast<double>(p_pos.y()) - static_cast<double>(q_pos.y())) *
            (static_cast<double>(p_pos.y()) - static_cast<double>(q_pos.y())) +
        (static_cast<double>(p_pos.z()) - static_cast<double>(q_pos.z())) *
            (static_cast<double>(p_pos.z()) - static_cast<double>(q_pos.z())));
    double C = 1.0/(M_PI * h*h*h);
    double u = r/h;
    double val = 0.0;
    if(u >= 2.0)
        return val;
    else if(u < 1.0 )
        val = -1.0/(h*h) * (3.0 - 9.0/4.0*u);
    else if(u >= 1.0 && u < 2.0)
        val = -3.0 / (4.0 * h * r) * sycl::pow(2.0 - u, 2.0);

    val *= C;
    return val;
}

////////////////////////////////////////////////////////////////////////////
// Boundary particle force
// http://iopscience.iop.org/0034-4885/68/8/R01/pdf/0034-4885_68_8_R01.pdf
////////////////////////////////////////////////////////////////////////////

double boundaryGamma(sycl::double3 p_pos, sycl::double3 k_pos,
                     sycl::double3 k_n, double h, double speed_sound)
{
    // Radial distance between p,q
    double r = sycl::sqrt(
        (static_cast<double>(p_pos.x()) - static_cast<double>(k_pos.x())) *
            (static_cast<double>(p_pos.x()) - static_cast<double>(k_pos.x())) +
        (static_cast<double>(p_pos.y()) - static_cast<double>(k_pos.y())) *
            (static_cast<double>(p_pos.y()) - static_cast<double>(k_pos.y())) +
        (static_cast<double>(p_pos.z()) - static_cast<double>(k_pos.z())) *
            (static_cast<double>(p_pos.z()) - static_cast<double>(k_pos.z())));
    // Distance to p normal to surface particle
    double y = sycl::sqrt(
        (static_cast<double>(p_pos.x()) - static_cast<double>(k_pos.x())) *
            (static_cast<double>(p_pos.x()) - static_cast<double>(k_pos.x())) *
            (static_cast<double>(k_n.x()) * static_cast<double>(k_n.x())) +
        (static_cast<double>(p_pos.y()) - static_cast<double>(k_pos.y())) *
            (static_cast<double>(p_pos.y()) - static_cast<double>(k_pos.y())) *
            (static_cast<double>(k_n.y()) * static_cast<double>(k_n.y())) +
        (static_cast<double>(p_pos.z()) - static_cast<double>(k_pos.z())) *
            (static_cast<double>(p_pos.z()) - static_cast<double>(k_pos.z())) *
            (static_cast<double>(k_n.z()) * static_cast<double>(k_n.z())));
    // Tangential distance
    double x = r-y;

    double u = y/h;
    double xi = (1-x/h)?x<h:0.0;
    double C = xi*2.0*0.02 * speed_sound * speed_sound / y;
    double val = 0.0;

    if(u > 0.0 && u < 2.0/3.0)
        val = 2.0/3.0;
    else if(u < 1.0 && u > 2.0/3.0 )
        val = (2*u - 3.0/2.0*u*u);
    else if (u < 2.0 && u > 1.0)
        val = 0.5*(2.0-u)*(2.0-u);
    else
        val = 0.0;

    val *= C;

    return val;
}

////////////////////////////////////////////////////////////////////////////
// Particle attribute computations
////////////////////////////////////////////////////////////////////////////

double computeDensity(sycl::double3 p_pos, sycl::double3 p_v,
                      sycl::double3 q_pos, sycl::double3 q_v, param *params)
{
    double v_x = (static_cast<double>(p_v.x()) - static_cast<double>(q_v.x()));
    double v_y = (static_cast<double>(p_v.y()) - static_cast<double>(q_v.y()));
    double v_z = (static_cast<double>(p_v.z()) - static_cast<double>(q_v.z()));

    double density = params->mass_particle * del_W(p_pos,q_pos,
                                                   params->smoothing_radius);
    double density_x =
        density * v_x *
        (static_cast<double>(p_pos.x()) - static_cast<double>(q_pos.x()));
    double density_y =
        density * v_y *
        (static_cast<double>(p_pos.y()) - static_cast<double>(q_pos.y()));
    double density_z =
        density * v_z *
        (static_cast<double>(p_pos.z()) - static_cast<double>(q_pos.z()));

    density = (density_x + density_y + density_z)*params->time_step;

    return density;
}


double computePressure(double p_density, param *params)
{
    double gam = 7.0;
    double B = params->rest_density * params->speed_sound*params->speed_sound / gam;
    double pressure =
        B * (sycl::pow((p_density / params->rest_density), gam) - 1.0);

    return pressure;
}


void updatePressures(fluid_particle *fluid_particles, param *params,
                     sycl::nd_item<3> item_ct1)
{
    int num_fluid_particles = params->number_fluid_particles;
    int i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);
    if (i >= num_fluid_particles) return;
    sycl::double3 p_pos = fluid_particles[i].pos;
    sycl::double3 p_v = fluid_particles[i].v;
    double density = fluid_particles[i].density;

    for(int j=0; j< num_fluid_particles; j++) {
        sycl::double3 q_pos = fluid_particles[j].pos;
        sycl::double3 q_v = fluid_particles[j].v;
        density += computeDensity(p_pos,p_v,q_pos,q_v, params);
    }
    fluid_particles[i].density = density;
    fluid_particles[i].pressure = computePressure(density, params);
}

sycl::double3 computeBoundaryAcceleration(sycl::double3 p_pos,
                                          sycl::double3 k_pos,
                                          sycl::double3 k_n, double h,
                                          double speed_sound)
{
    sycl::double3 p_a;
    double bGamma = boundaryGamma(p_pos,k_pos,k_n,h,speed_sound);
    p_a.x() = bGamma * static_cast<double>(k_n.x());
    p_a.y() = bGamma * static_cast<double>(k_n.y());
    p_a.z() = bGamma * static_cast<double>(k_n.z());

    return p_a;
}

sycl::double3 computeAcceleration(sycl::double3 p_pos, sycl::double3 p_v,
                                  double p_density, double p_pressure,
                                  sycl::double3 q_pos, sycl::double3 q_v,
                                  double q_density, double q_pressure,
                                  const param *const params)
{
    sycl::double3 a;
    double accel;
    double h = params->smoothing_radius;
    double alpha = params->alpha;
    double speed_sound = params->speed_sound;
    double mass_particle = params->mass_particle;
    double surface_tension = params->surface_tension;

    // Pressure force
    accel = (p_pressure/(p_density*p_density) + q_pressure/(q_density*q_density))
            * mass_particle * del_W(p_pos,q_pos,h);
    a.x() = -accel *
            (static_cast<double>(p_pos.x()) - static_cast<double>(q_pos.x()));
    a.y() = -accel *
            (static_cast<double>(p_pos.y()) - static_cast<double>(q_pos.y()));
    a.z() = -accel *
            (static_cast<double>(p_pos.z()) - static_cast<double>(q_pos.z()));

    // Viscosity force
    double VdotR =
        (static_cast<double>(p_v.x()) - static_cast<double>(q_v.x())) *
            (static_cast<double>(p_pos.x()) - static_cast<double>(q_pos.x())) +
        (static_cast<double>(p_v.y()) - static_cast<double>(q_v.y())) *
            (static_cast<double>(p_pos.y()) - static_cast<double>(q_pos.y())) +
        (static_cast<double>(p_v.z()) - static_cast<double>(q_v.z())) *
            (static_cast<double>(p_pos.z()) - static_cast<double>(q_pos.z()));
    if(VdotR < 0.0)
    {
        double nu = 2.0 * alpha * h * speed_sound / (p_density + q_density);
        double r2 =
            (static_cast<double>(p_pos.x()) - static_cast<double>(q_pos.x())) *
                (static_cast<double>(p_pos.x()) -
                 static_cast<double>(q_pos.x())) +
            (static_cast<double>(p_pos.y()) - static_cast<double>(q_pos.y())) *
                (static_cast<double>(p_pos.y()) -
                 static_cast<double>(q_pos.y())) +
            (static_cast<double>(p_pos.z()) - static_cast<double>(q_pos.z())) *
                (static_cast<double>(p_pos.z()) -
                 static_cast<double>(q_pos.z()));
        double eps = h/10.0;
        double stress = nu * VdotR / (r2 + eps*h*h);
        accel = mass_particle * stress * del_W(p_pos, q_pos, h);
        a.x() += accel * (static_cast<double>(p_pos.x()) -
                          static_cast<double>(q_pos.x()));
        a.y() += accel * (static_cast<double>(p_pos.y()) -
                          static_cast<double>(q_pos.y()));
        a.z() += accel * (static_cast<double>(p_pos.z()) -
                          static_cast<double>(q_pos.z()));
    }

    //Surface tension
    // BT 07 http://cg.informatik.uni-freiburg.de/publications/2011_GRAPP_airBubbles.pdf
    accel = surface_tension * W(p_pos,q_pos,h);
    a.x() += accel *
             (static_cast<double>(p_pos.x()) - static_cast<double>(q_pos.x()));
    a.y() += accel *
             (static_cast<double>(p_pos.y()) - static_cast<double>(q_pos.y()));
    a.z() += accel *
             (static_cast<double>(p_pos.z()) - static_cast<double>(q_pos.z()));

    return a;
}

// Update particle acclerations

void updateAccelerationsFP(fluid_particle *fluid_particles, param *params,
                           sycl::nd_item<3> item_ct1)
{
    int num_fluid_particles = params->number_fluid_particles;

    int i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);
    if (i >= num_fluid_particles) return;

        double ax = 0.0;
        double ay = 0.0;
        double az = -9.8;

    sycl::double3 p_pos = fluid_particles[i].pos;
    sycl::double3 p_v = fluid_particles[i].v;
        double p_density = fluid_particles[i].density;
        double p_pressure = fluid_particles[i].pressure;

        for(int j=0; j<num_fluid_particles; j++) {
            if (i!=j) {
            sycl::double3 q_pos = fluid_particles[j].pos;
            sycl::double3 q_v = fluid_particles[j].v;
                double q_density = fluid_particles[j].density;
                double q_pressure = fluid_particles[j].pressure;
            sycl::double3 tmp_a =
                computeAcceleration(p_pos, p_v, p_density, p_pressure, q_pos,
                                    q_v, q_density, q_pressure, params);

            ax += static_cast<double>(tmp_a.x());
            ay += static_cast<double>(tmp_a.y());
            az += static_cast<double>(tmp_a.z());
            }
        }

    fluid_particles[i].a.x() = ax;
    fluid_particles[i].a.y() = ay;
    fluid_particles[i].a.z() = az;
}


void updateAccelerationsBP(fluid_particle *fluid_particles,
                         boundary_particle *boundary_particles, 
                         param *params,
                         sycl::nd_item<3> item_ct1)
{
    int num_fluid_particles = params->number_fluid_particles;
    int num_boundary_particles = params->number_boundary_particles;
    int i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);
    if (i >= num_fluid_particles) return;

    double ax = static_cast<double>(fluid_particles[i].a.x());
    double ay = static_cast<double>(fluid_particles[i].a.y());
    double az = static_cast<double>(fluid_particles[i].a.z());
    sycl::double3 p_pos = fluid_particles[i].pos;

    for (int j=0; j<num_boundary_particles; j++) {
        sycl::double3 k_pos = boundary_particles[j].pos;
        sycl::double3 k_n = boundary_particles[j].n;
        sycl::double3 tmp_a = computeBoundaryAcceleration(
            p_pos, k_pos, k_n, params->smoothing_radius, params->speed_sound);
        ax += static_cast<double>(tmp_a.x());
        ay += static_cast<double>(tmp_a.y());
        az += static_cast<double>(tmp_a.z());
    }

    fluid_particles[i].a.x() = ax;
    fluid_particles[i].a.y() = ay;
    fluid_particles[i].a.z() = az;
}

// Update particle positions
// Leap Frog integration with v(t+1) estimated

void updatePositions(fluid_particle *fluid_particles, param *params,
                     sycl::nd_item<3> item_ct1)
{
    double dt = params->time_step;

    int num_fluid_particles = params->number_fluid_particles;
    int i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);
    if (i >= num_fluid_particles) return;

    // Velocity at t + dt/2
    sycl::double3 v_half = fluid_particles[i].v_half;
    sycl::double3 v = fluid_particles[i].v;
    sycl::double3 pos = fluid_particles[i].pos;
    sycl::double3 a = fluid_particles[i].a;

    v_half.x() =
        static_cast<double>(v_half.x()) + dt * static_cast<double>(a.x());
    v_half.y() =
        static_cast<double>(v_half.y()) + dt * static_cast<double>(a.y());
    v_half.z() =
        static_cast<double>(v_half.z()) + dt * static_cast<double>(a.z());

    // Velocity at t + dt, must estimate for foce calc
    v.x() = static_cast<double>(v_half.x()) +
            static_cast<double>(a.x()) * (dt / 2.0);
    v.y() = static_cast<double>(v_half.y()) +
            static_cast<double>(a.y()) * (dt / 2.0);
    v.z() = static_cast<double>(v_half.z()) +
            static_cast<double>(a.z()) * (dt / 2.0);

    // Position at time t + dt
    pos.x() =
        static_cast<double>(pos.x()) + dt * static_cast<double>(v_half.x());
    pos.y() =
        static_cast<double>(pos.y()) + dt * static_cast<double>(v_half.y());
    pos.z() =
        static_cast<double>(pos.z()) + dt * static_cast<double>(v_half.z());

    fluid_particles[i].v_half = v_half;
    fluid_particles[i].v      = v;
    fluid_particles[i].pos    = pos;
}

// Seed simulation with Euler step v(t-dt/2) needed by leap frog integrator
// Should calculate all accelerations but assuming just g simplifies acc port
void eulerStart(fluid_particle* fluid_particles,
                boundary_particle* boundary_particles, param *params)
{
    // Set V (t0 - dt/2)
    double dt_half = params->time_step/2.0;

    for(int i=0; i<params->number_fluid_particles; i++)
    {
        // Velocity at t + dt/2
        sycl::double3 v = fluid_particles[i].v;
        sycl::double3 v_half;

        v_half.x() = static_cast<double>(v.x());
        v_half.y() = static_cast<double>(v.y());
        v_half.z() = static_cast<double>(v.z()) - params->g * dt_half;

        fluid_particles[i].v_half = v_half;
    }
}

// Initialize particles
void initParticles(fluid_particle** fluid_particles, boundary_particle** boundary_particles,
                   AABB* water, AABB* boundary, param* params)
{
    // Allocate fluid particles array
    *fluid_particles = (fluid_particle*) malloc(params->number_fluid_particles * sizeof(fluid_particle));
    // Allocate boundary particles array
    *boundary_particles = (boundary_particle*) malloc(params->number_boundary_particles * sizeof(boundary_particle));

    double spacing = params->spacing_particle;

    // Initialize particle values
    for(int i=0; i<params->number_fluid_particles; i++) {
        (*fluid_particles)[i].a.x() = 0.0;
        (*fluid_particles)[i].a.y() = 0.0;
        (*fluid_particles)[i].a.z() = 0.0;
        (*fluid_particles)[i].v.x() = 0.0;
        (*fluid_particles)[i].v.y() = 0.0;
        (*fluid_particles)[i].v.z() = 0.0;
        (*fluid_particles)[i].density = params->rest_density;
    }

    // Place particles inside bounding volume
    double x,y,z;
    int i = 0;
    for(z=water->min_z; z<=water->max_z; z+=spacing) {
        for(y=water->min_y; y<=water->max_y; y+=spacing) {
            for(x=water->min_x; x<=water->max_x; x+=spacing) {
                if(i < params->number_fluid_particles) {
                    (*fluid_particles)[i].pos.x() = x;
                    (*fluid_particles)[i].pos.y() = y;
                    (*fluid_particles)[i].pos.z() = z;
                    i++;
                }
            }
        }
    }
    params->number_fluid_particles = i;

    // Construct bounding box
    constructBoundaryBox(*boundary_particles, boundary, params);
}

void initParams(AABB* water_volume, AABB* boundary_volume, param* params)
{
    // Boundary box
    boundary_volume->min_x = 0.0;
    boundary_volume->max_x = 1.1;
    boundary_volume->min_y = 0.0;
    boundary_volume->max_y = 1.1;
    boundary_volume->min_z = 0.0;
    boundary_volume->max_z = 1.1;

    // water volume
    water_volume->min_x = 0.1;
    water_volume->max_x = 0.5;
    water_volume->min_y = 0.1;
    water_volume->max_y = 0.5;
    water_volume->min_z = 0.08;
    water_volume->max_z = 0.8;

    // Simulation parameters
    params->number_fluid_particles = 2048;
    params->rest_density = 1000.0;
    params->g = 9.8;
    params->alpha = 0.02;
    params->surface_tension =  0.01;
    params->number_steps = 500; // reduce from 5000
    params->time_step = 0.00035;

    // Mass of each particle
    double volume = (water_volume->max_x - water_volume->min_x)
                  * (water_volume->max_y - water_volume->min_y)
                  * (water_volume->max_z - water_volume->min_z);
    params->mass_particle = params->rest_density * (volume/params->number_fluid_particles);

    // Cube calculated spacing
    params->spacing_particle =
        pow(volume / params->number_fluid_particles, 1.0 / 3.0);

    // Smoothing radius, h
    params->smoothing_radius = params->spacing_particle;

    // Boundary particles
    int num_x = ceil((boundary_volume->max_x - boundary_volume->min_x) /
                     params->spacing_particle);
    int num_y = ceil((boundary_volume->max_y - boundary_volume->min_y) /
                     params->spacing_particle);
    int num_z = ceil((boundary_volume->max_z - boundary_volume->min_z) /
                     params->spacing_particle);
    int num_boundary_particles = (2 * num_x * num_z) + (2 * num_y * num_z) + (2* num_y * num_z);
    params->number_boundary_particles = num_boundary_particles;

    // Total number of particles
    params->number_particles = params->number_boundary_particles + params->number_fluid_particles;

    // Number of steps before frame needs to be written for 30 fps
    params->steps_per_frame = (int)(1.0/(params->time_step*30.0));

    // Calculate speed of sound for simulation
    double max_height = water_volume->max_y;
    double max_velocity = sqrt(2.0 * params->g * max_height);
    params->speed_sound = max_velocity / sqrt(0.01);

    // Minimum stepsize from Courant-Friedrichs-Lewy condition
    double recomend_step = 0.4 * params->smoothing_radius / (params->speed_sound
                                                         * (1+ 0.6*params->alpha));
    printf("Using time step: %f, Minimum recomended %f\n",params->time_step, recomend_step);
}

void finalizeParticles(fluid_particle *fluid_particles, boundary_particle *boundary_particles)
{
    free(fluid_particles);
    free(boundary_particles);
}

int main(int argc, char *argv[])
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    param params;
    AABB water_volume;
    AABB boundary_volume;
    fluid_particle *fluid_particles = NULL;
    boundary_particle *boundary_particles = NULL;
    initParams(&water_volume, &boundary_volume, &params);

    initParticles(&fluid_particles, &boundary_particles, &water_volume,
                  &boundary_volume, &params);

    eulerStart(fluid_particles, boundary_particles, &params);

    int num_fluid_particles = params.number_fluid_particles;
    int num_boundary_particles = params.number_boundary_particles;

    fluid_particle *d_fluid_particles;
    boundary_particle *d_boundary_particles;
    param *d_params;
    d_fluid_particles =
        sycl::malloc_device<fluid_particle>(num_fluid_particles, q_ct1);
    d_boundary_particles =
        sycl::malloc_device<boundary_particle>(num_boundary_particles, q_ct1);
    d_params = sycl::malloc_device<param>(1, q_ct1);
    q_ct1
        .memcpy(d_fluid_particles, fluid_particles,
                num_fluid_particles * sizeof(fluid_particle))
        .wait();
    q_ct1
        .memcpy(d_boundary_particles, boundary_particles,
                num_boundary_particles * sizeof(boundary_particle))
        .wait();
    q_ct1.memcpy(d_params, &params, sizeof(param)).wait();

    sycl::range<3> block1D(256, 1, 1);
    sycl::range<3> grid1D_FP((num_fluid_particles + 255) / 256, 1, 1);
    sycl::range<3> grid1D_BP((num_boundary_particles + 255) / 256, 1, 1);

    // Main simulation loop
    for(int n=0; n<params.number_steps; n++) {
        q_ct1.submit([&](sycl::handler &cgh) {
            auto dpct_global_range =
                sycl::range<3>(grid1D_FP) * sycl::range<3>(block1D);
            auto dpct_local_range = sycl::range<3>(block1D);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                 dpct_global_range.get(1),
                                                 dpct_global_range.get(0)),
                                  sycl::range<3>(dpct_local_range.get(2),
                                                 dpct_local_range.get(1),
                                                 dpct_local_range.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    updatePressures(d_fluid_particles, d_params, item_ct1);
                });
        });
        q_ct1.submit([&](sycl::handler &cgh) {
            auto dpct_global_range =
                sycl::range<3>(grid1D_FP) * sycl::range<3>(block1D);
            auto dpct_local_range = sycl::range<3>(block1D);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                 dpct_global_range.get(1),
                                                 dpct_global_range.get(0)),
                                  sycl::range<3>(dpct_local_range.get(2),
                                                 dpct_local_range.get(1),
                                                 dpct_local_range.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    updateAccelerationsFP(d_fluid_particles, d_params,
                                          item_ct1);
                });
        });
        q_ct1.submit([&](sycl::handler &cgh) {
            auto dpct_global_range =
                sycl::range<3>(grid1D_BP) * sycl::range<3>(block1D);
            auto dpct_local_range = sycl::range<3>(block1D);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                 dpct_global_range.get(1),
                                                 dpct_global_range.get(0)),
                                  sycl::range<3>(dpct_local_range.get(2),
                                                 dpct_local_range.get(1),
                                                 dpct_local_range.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    updateAccelerationsBP(d_fluid_particles,
                                          d_boundary_particles, d_params,
                                          item_ct1);
                });
        });
        q_ct1.submit([&](sycl::handler &cgh) {
            auto dpct_global_range =
                sycl::range<3>(grid1D_FP) * sycl::range<3>(block1D);
            auto dpct_local_range = sycl::range<3>(block1D);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                 dpct_global_range.get(1),
                                                 dpct_global_range.get(0)),
                                  sycl::range<3>(dpct_local_range.get(2),
                                                 dpct_local_range.get(1),
                                                 dpct_local_range.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    updatePositions(d_fluid_particles, d_params, item_ct1);
                });
        });
        //cudaDeviceSynchronize();
    }
    q_ct1
        .memcpy(fluid_particles, d_fluid_particles,
                num_fluid_particles * sizeof(fluid_particle))
        .wait();

    writeFile(fluid_particles, &params);

    finalizeParticles(fluid_particles, boundary_particles);
    sycl::free(d_fluid_particles, q_ct1);
    sycl::free(d_boundary_particles, q_ct1);
    sycl::free(d_params, q_ct1);
    return 0;
}

