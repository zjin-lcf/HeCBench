#include <cmath>
#include <cfloat>  // FLT_MAX
#include "bude.h"

#define ZERO    0.0f
#define QUARTER 0.25f
#define HALF    0.5f
#define ONE     1.0f
#define TWO     2.0f
#define FOUR    4.0f
#define CNSTNT 45.0f

// Energy evaluation parameters
#define HBTYPE_F 70
#define HBTYPE_E 69
#define HARDNESS 38.0f
#define NPNPDIST  5.5f
#define NPPDIST   1.0f

SYCL_EXTERNAL
__attribute__ ((always_inline))
void fasten_main(
    sycl::nd_item<1> &item,
    FFParams *local_forcefield,
    size_t wgSize,
    size_t ntypes,
    size_t nposes,
    size_t natlig,
    size_t natpro,
    Atom *protein_molecule,
    Atom *ligand_molecule,
    float *transforms_0,
    float *transforms_1,
    float *transforms_2,
    float *transforms_3,
    float *transforms_4,
    float *transforms_5,
    FFParams *forcefield,
    float *etotals)
{
  const size_t lid = item.get_local_id(0);
  const size_t gid = item.get_group(0);
  const size_t lrange = item.get_local_range(0);

  float etot[NUM_TD_PER_THREAD];
  sycl::float3 lpos[NUM_TD_PER_THREAD];
  sycl::float4 transform[NUM_TD_PER_THREAD][3];

  size_t ix = gid * lrange * NUM_TD_PER_THREAD + lid;
  ix = ix < nposes ? ix : nposes - NUM_TD_PER_THREAD;

  for (int i = lid; i < ntypes; i += lrange)
    local_forcefield[i] = forcefield[i];
  //if (ix < ntypes) local_forcefield[ix] = forcefield[ix];

  // Compute transformation matrix to private memory
  for (size_t i = 0; i < NUM_TD_PER_THREAD; i++) {
    size_t index = ix + i * lrange;

    const float sx = sycl::sin(transforms_0[index]);
    const float cx = sycl::cos(transforms_0[index]);
    const float sy = sycl::sin(transforms_1[index]);
    const float cy = sycl::cos(transforms_1[index]);
    const float sz = sycl::sin(transforms_2[index]);
    const float cz = sycl::cos(transforms_2[index]);

    transform[i][0].x() = cy * cz;
    transform[i][0].y() = sx * sy * cz - cx * sz;
    transform[i][0].z() = cx * sy * cz + sx * sz;
    transform[i][0].w() = transforms_3[index];
    transform[i][1].x() = cy * sz;
    transform[i][1].y() = sx * sy * sz + cx * cz;
    transform[i][1].z() = cx * sy * sz - sx * cz;
    transform[i][1].w() = transforms_4[index];
    transform[i][2].x() = -sy;
    transform[i][2].y() = sx * cy;
    transform[i][2].z() = cx * cy;
    transform[i][2].w() = transforms_5[index];

    etot[i] = ZERO;
  }

  item.barrier(sycl::access::fence_space::local_space);

  // Loop over ligand atoms
  size_t il = 0;
  do {
    // Load ligand atom data
    const Atom l_atom = ligand_molecule[il];
    const FFParams l_params = local_forcefield[l_atom.type];
    const bool lhphb_ltz = l_params.hphb < ZERO;
    const bool lhphb_gtz = l_params.hphb > ZERO;

    const sycl::float4 linitpos(l_atom.x, l_atom.y, l_atom.z, ONE);
    for (size_t i = 0; i < NUM_TD_PER_THREAD; i++) {
      // Transform ligand atom
      lpos[i].x() = transform[i][0].w() +
        linitpos.x() * transform[i][0].x() +
        linitpos.y() * transform[i][0].y() +
        linitpos.z() * transform[i][0].z();
      lpos[i].y() = transform[i][1].w() +
        linitpos.x() * transform[i][1].x() +
        linitpos.y() * transform[i][1].y() +
        linitpos.z() * transform[i][1].z();
      lpos[i].z() = transform[i][2].w() +
        linitpos.x() * transform[i][2].x() +
        linitpos.y() * transform[i][2].y() +
        linitpos.z() * transform[i][2].z();
    }

    // Loop over protein atoms
    size_t ip = 0;
    do {
      // Load protein atom data
      const Atom p_atom = protein_molecule[ip];
      const FFParams p_params = local_forcefield[p_atom.type];

      const float radij = p_params.radius + l_params.radius;
      const float r_radij = 1.f / (radij);

      const float elcdst = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? FOUR : TWO;
      const float elcdst1 = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? QUARTER : HALF;
      const bool type_E = ((p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E));

      const bool phphb_ltz = p_params.hphb < ZERO;
      const bool phphb_gtz = p_params.hphb > ZERO;
      const bool phphb_nz = p_params.hphb != ZERO;
      const float p_hphb = p_params.hphb * (phphb_ltz && lhphb_gtz ? -ONE : ONE);
      const float l_hphb = l_params.hphb * (phphb_gtz && lhphb_ltz ? -ONE : ONE);
      const float distdslv = (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST) : (lhphb_ltz ? NPPDIST : -FLT_MAX));
      const float r_distdslv = 1.f / (distdslv);

      const float chrg_init = l_params.elsc * p_params.elsc;
      const float dslv_init = p_hphb + l_hphb;

      for (size_t i = 0; i < NUM_TD_PER_THREAD; i++) {
        // Calculate distance between atoms
        const float x = lpos[i].x() - p_atom.x;
        const float y = lpos[i].y() - p_atom.y;
        const float z = lpos[i].z() - p_atom.z;

        const float distij = sycl::sqrt(x * x + y * y + z * z);

        // Calculate the sum of the sphere radii
        const float distbb = distij - radij;
        const bool zone1 = (distbb < ZERO);

        // Calculate steric energy
        etot[i] += (ONE - (distij * r_radij)) * (zone1 ? 2 * HARDNESS : ZERO);

        // Calculate formal and dipole charge interactions
        float chrg_e = chrg_init * ((zone1 ? 1 : (ONE - distbb * elcdst1)) * (distbb < elcdst ? 1 : ZERO));
        const float neg_chrg_e = -sycl::fabs(chrg_e);
        chrg_e = type_E ? neg_chrg_e : chrg_e;
        etot[i] += chrg_e * CNSTNT;

        // Calculate the two cases for Nonpolar-Polar repulsive interactions
        const float coeff = (ONE - (distbb * r_distdslv));
        float dslv_e = dslv_init * ((distbb < distdslv && phphb_nz) ? 1 : ZERO);
        dslv_e *= (zone1 ? 1 : coeff);
        etot[i] += dslv_e;
      }
    } while (++ip < natpro); // loop over protein atoms
  } while (++il < natlig); // loop over ligand atoms

  // Write results
  const size_t td_base = gid * lrange * NUM_TD_PER_THREAD + lid;

  if (td_base < nposes) {
    for (size_t i = 0; i < NUM_TD_PER_THREAD; i++) {
      etotals[td_base + i * lrange] = etot[i] * HALF;
    }
  }
}
