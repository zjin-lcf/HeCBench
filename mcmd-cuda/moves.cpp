#include <string>
#include <algorithm>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <map>
#include <string>
#include <stdlib.h>

// for debuggin
void printEnergies(System &system) {
  int debug=0;
  if (debug) {
    printf("rd: %f es: %f pol: %f tot: %f\n", system.stats.rd.value, system.stats.es.value, system.stats.polar.value, system.stats.potential.value);
  }
}

int getRandomMovableMolecule(System &system) {
  // re-do list of movable IDs only if uVT (the vector is defined in main.cpp)
  if (system.constants.ensemble == ENSEMBLE_UVT) {
    system.stats.movids.clear();
    // get a list of movable IDs
    for (int i=0; i<system.molecules.size(); i++)
      if (!system.molecules[i].frozen)
        system.stats.movids.push_back(i);
  }
  // get random id
  return system.stats.movids[floor(getrand()*(double)system.stats.count_movables)];
}

void translate(System &system, int molid) {
  double randx,randy,randz;
  randx = system.constants.displace_factor * (getrand()*2-1);
  randy = system.constants.displace_factor * (getrand()*2-1);
  randz = system.constants.displace_factor * (getrand()*2-1);

  system.molecules[molid].com[0] += randx;
  system.molecules[molid].com[1] += randy;
  system.molecules[molid].com[2] += randz;

  for (int i=0; i<system.molecules[molid].atoms.size(); i++) {
    system.molecules[molid].atoms[i].pos[0] += randx;
    system.molecules[molid].atoms[i].pos[1] += randy;
    system.molecules[molid].atoms[i].pos[2] += randz;
  }
} // end translate()

void rotate(System &system, int molid) {
  system.checkpoint("doing a rotation move.");
  double com[3], randangle;
  int DIM, i, n;

  for (DIM=0; DIM<3; DIM++) { // 3 planes of rotation
    // 1) GET RANDOM ANGLE
    randangle = system.constants.rotate_angle_factor*getrand();

    // 2) SAVE CURRENT COM
    for (n=0; n<3; n++) com[n] = system.molecules[molid].com[n];

    // 3) MOVE MOLECULE TO ORIGIN TEMPORARILY
    for (i=0; i<system.molecules[molid].atoms.size(); i++) {
      for (n=0; n<3; n++) {
        system.molecules[molid].atoms[i].pos[n] -= com[n];
      }
    }

    // 4) ROTATE THE MOLECULE ABOUT ORIGIN IN ALL 3 DIMS
    for (i=0; i<system.molecules[molid].atoms.size(); i++) {
      double* rotated = rotatePoint(system, system.molecules[molid].atoms[i].pos[0], system.molecules[molid].atoms[i].pos[1], system.molecules[molid].atoms[i].pos[2], DIM, randangle);
      system.molecules[molid].atoms[i].pos[0] = rotated[0];
      system.molecules[molid].atoms[i].pos[1] = rotated[1];
      system.molecules[molid].atoms[i].pos[2] = rotated[2];
    } // end for i

    // 5) MOVE MOLECULE BACK TO COM POSITION, BUT ROTATED
    for (i=0; i<system.molecules[molid].atoms.size(); i++)
      for (n=0; n<3; n++)
        system.molecules[molid].atoms[i].pos[n] += com[n];

  } // end 3D
} // end rotate();


/* (RE)DEFINE THE BOX LENGTHS */
void defineBox(System &system) { // takes input in A
  // easy 90 90 90 systems
  if (system.pbc.alpha == 90 && system.pbc.beta == 90 && system.pbc.gamma == 90) {
    // assumes x_length, y_length, z_length are defined already in system.
    // i.e. the volume-change function does that before calling this function

    system.pbc.basis[0][0] = system.pbc.x_length;
    system.pbc.basis[1][1] = system.pbc.y_length;
    system.pbc.basis[2][2] = system.pbc.z_length;

    system.pbc.x_max = system.pbc.x_length/2.0;
    system.pbc.x_min = -system.pbc.x_max;
    system.pbc.y_max = system.pbc.y_length/2.0;
    system.pbc.y_min = -system.pbc.y_max;
    system.pbc.z_max = system.pbc.z_length/2.0;
    system.pbc.z_min = -system.pbc.z_max;

    system.pbc.calcVolume();
    system.pbc.calcRecip();
    system.pbc.calcCutoff();
    system.constants.ewald_alpha = 3.5/system.pbc.cutoff; // update ewald_alpha if we have a vol change
    // no need for vertices and planes for 90/90/90

  }
  // universal definitions
  else {
    // this could forseeably be a problem if, for some weird reason, someone wants to do NPT with a weird box.
    system.pbc.calcVolume();
    system.pbc.calcRecip();
    system.pbc.calcCutoff();
    system.constants.ewald_alpha = 3.5/system.pbc.cutoff;
    system.pbc.calcBoxVertices();
    system.pbc.calcPlanes();
  }
}

/* CHANGE VOLUME OF SYSTEM BY BOLTZMANN PROB -- FOR NPT */
void changeVolumeMove(System &system) {
  system.stats.volume_attempts++;
  // generate small random distance change for volume adjustment
  double ranf = getrand();  // for boltz check
  double ranv = getrand(); // for volume change
  double old_energy, new_energy;
  double new_com[3], old_com[3], delta_pos[3];
  int i,j,n;

  old_energy=system.stats.potential.value; //getTotalPotential(system);

  system.pbc.old_volume = system.pbc.volume;

  // change the volume to test new energy.
  double new_volume = exp(log(system.pbc.volume) + (ranv-0.5)*system.constants.volume_change);// mpmc default = 2.5
  double basis_scale_factor = pow(new_volume/system.pbc.volume, 1.0/3.0);
  system.pbc.x_length *= basis_scale_factor;
  system.pbc.y_length *= basis_scale_factor;
  system.pbc.z_length *= basis_scale_factor;
  defineBox(system);

  // scale molecule positions
  for (i=0; i<system.molecules.size(); i++) {
    for (n=0; n<3; n++) {
      old_com[n] = system.molecules[i].com[n];
      new_com[n] = system.molecules[i].com[n]*basis_scale_factor;
      delta_pos[n] = new_com[n] - old_com[n];
    }
    // move atoms one by one
    for (j=0; j<system.molecules[i].atoms.size(); j++) {
      for (n=0; n<3; n++) {
        system.molecules[i].atoms[j].pos[n] += delta_pos[n];
      }
    }
    system.molecules[i].calc_center_of_mass();
  }

  new_energy=getTotalPotential(system);

  double boltzmann_factor = get_boltzmann_factor(system, old_energy, new_energy, MOVETYPE_VOLUME);

  if (ranf < boltzmann_factor && system.constants.iter_success == 0) {
    // accept move
    system.stats.volume_change_accepts++;
    system.stats.MCmoveAccepted = true;
    printEnergies(system);
  } else {
    system.constants.iter_success = 0;
    // reject move (move volume back)
    system.pbc.x_length /= basis_scale_factor;
    system.pbc.y_length /= basis_scale_factor;
    system.pbc.z_length /= basis_scale_factor;
    defineBox(system);

    // move molecules back
    for (i=0; i<system.molecules.size(); i++) {
      for (n=0; n<3; n++) {
        old_com[n] = system.molecules[i].com[n];
        new_com[n] = system.molecules[i].com[n]/basis_scale_factor;
        delta_pos[n] = new_com[n] - old_com[n];
      }
      for (j=0; j<system.molecules[i].atoms.size(); j++) {
        for (n=0; n<3; n++)
          system.molecules[i].atoms[j].pos[n] += delta_pos[n];
      }
      system.molecules[i].calc_center_of_mass();
    }
  }
}


/* ADD A MOLECULE */
void addMolecule(System &system) {
  system.checkpoint("starting addMolecule");
  system.stats.insert_attempts++;
  int protoid;

  // get current energy.
  double old_potential = system.stats.potential.value; //getTotalPotential(system);

  // select a random prototype molecule
  if (system.proto.size() == 1) protoid=0;
  else {
    protoid = (rand() % (int)(system.proto.size()));
  }
  system.molecules.push_back(system.proto[protoid]);
  system.constants.currentprotoid = protoid; // for getting boltz factor later.
  system.stats.count_movables += 1;

  //id in vector INDEX. .ID is PDB ID
  int last_molecule_id = (int)system.molecules.size()-1;
  system.molecules[last_molecule_id].PDBID = system.molecules[last_molecule_id -1].PDBID + 1; // .PDBID is the last one +1
  int last_molecule_PDBID = system.molecules[last_molecule_id].PDBID;

  // set mol_id variable on individual atoms
  for (int i=0; i<system.molecules[last_molecule_id].atoms.size(); i++) {
    system.molecules[last_molecule_id].atoms[i].mol_PDBID = last_molecule_PDBID;
    system.molecules[last_molecule_id].atoms[i].PDBID = system.constants.total_atoms + 1;
    system.constants.total_atoms += 1;
  }

  // for random placement in the unit cell
  double randn[3];
  int p,q; //,n;
  double move[3];
  for (p=0; p<3; p++)
    randn[p] = 0.5 - getrand();
  for (p=0; p<3; p++) {
    move[p]=0;
    for (q=0; q<3; q++)
      move[p] += system.pbc.basis[q][p]*randn[q];
  }

  // translate the new molecule's atoms to random place.
  for (int i=0; i<system.molecules[last_molecule_id].atoms.size(); i++) {
    for (int n=0; n<3; n++)
      system.molecules[last_molecule_id].atoms[i].pos[n] += move[n];
  }

  // rotate the molecule here by random amount.
  rotate(system, last_molecule_id);

  // **IMPORTANT: MAKE SURE THE MOLECULE IS IN THE BOX**
  checkInTheBox(system, last_molecule_id);

  // FULLY DONE ADDING MOLECULE TO SYSTEM IN PLACE. NOW GET NEW ENERGY
  double new_potential = getTotalPotential(system);

  // BOLTZMANN ACCEPT OR REJECT
  double boltz_factor = get_boltzmann_factor(system, old_potential, new_potential, MOVETYPE_INSERT);

  double ranf = getrand();
  if (ranf < boltz_factor && system.constants.iter_success ==0) {
    system.stats.insert_accepts++; //accept (keeps new molecule)
    system.stats.MCmoveAccepted = true;
    if (system.constants.mode == "md") {
      system.molecules[last_molecule_id].calc_inertia(); // need this for MD
    }
    printEnergies(system);
  } else {
    system.constants.iter_success = 0;
    // remove the new molecule.
    system.molecules.pop_back();
    system.constants.total_atoms -= (int)system.proto[protoid].atoms.size();
    system.stats.count_movables--;
  }
  system.checkpoint("done with addMolecule");
}


/* REMOVE A MOLECULE */
void removeMolecule(System &system) {
  system.checkpoint("starting removeMolecule");

  if (system.stats.count_movables == 0 ||
      (system.constants.no_zero_option && system.stats.count_movables == 1)) {
    return; // skip if no molecules are there to be removed, or if 1 left and no-zero option on.
  }
  system.stats.remove_attempts++;

  // get original energy.
  double old_potential = system.stats.potential.value; //getTotalPotential(system);

  system.checkpoint("getting random movable.");
  int randm = getRandomMovableMolecule(system);
  system.checkpoint("random movable selected.");

  // save a copy of this moleucule.
  Molecule tmp_molecule = system.molecules[randm];

  // delete the molecule
  system.molecules.erase(system.molecules.begin() + randm);
  system.stats.count_movables--;
  system.constants.total_atoms -= tmp_molecule.atoms.size(); //(int)system.proto[protoid].atoms.size();

  // get new energy
  double new_potential = getTotalPotential(system);

  // calculate BOLTZMANN FACTOR
  double boltz_factor = get_boltzmann_factor(system, old_potential, new_potential, MOVETYPE_REMOVE);

  // accept or reject
  double ranf = getrand();
  if (ranf < boltz_factor && system.constants.iter_success == 0) {
    system.stats.remove_accepts++;
    system.stats.MCmoveAccepted = true;
    printEnergies(system);
  } else {
    system.constants.iter_success = 0;
    // put the molecule back.
    system.molecules.push_back(tmp_molecule);
    system.constants.total_atoms += tmp_molecule.atoms.size(); //(int)system.proto.atoms.size();
    system.stats.count_movables++;
  }   // end boltz accept/reject
}


/* DISPLACE (TRANSLATE AND ROTATE) */
void displaceMolecule(System &system) {
  double tmpcom[3], d[3]; //, rsq;

  if (system.stats.count_movables == 0) return; // skip if no sorbate molecules are in the cell.
  system.stats.displace_attempts++;
  int randm = getRandomMovableMolecule(system);
  system.checkpoint("Got the random molecule.");

  for (int n=0; n<3; n++) tmpcom[n] = system.molecules[randm].com[n];

  double old_V=0.0;
  double new_V=0.0;

  // first calculate the system's current potential energy
  old_V = system.stats.potential.value; //getTotalPotential(system);

  // save a temporary copy of molecule to go back if needed
  Molecule tmp_molecule = system.molecules[randm];

  // do rotation AND translation
  // TRANSLATE
  system.checkpoint("doing translate move.");
  translate(system, randm);
  // ROTATION
  if (system.molecules[randm].atoms.size() > 1 && system.constants.rotate_option) { // try rotation
    rotate(system, randm);
  } // end rotation option

  // check P.B.C. (move the molecule back in the box if needed)
  checkInTheBox(system, randm);

  new_V = getTotalPotential(system);

  // now accept or reject the move based on Boltzmann probability
  double boltzmann_factor = get_boltzmann_factor(system, old_V, new_V, MOVETYPE_DISPLACE);

  // make ranf for probability pick
  double ranf = getrand(); // a value between 0 and 1

  // apply selection Frenkel Smit p. 30
  if (ranf < boltzmann_factor && system.constants.iter_success == 0) {
    system.stats.displace_accepts++;
    system.stats.MCmoveAccepted = true;

    // for MC efficiency measurement (Frenkel p44)
    for (int n=0; n<3; n++) d[n] = (system.molecules[randm].com[n] - tmpcom[n]);
    system.stats.MCeffRsq += dddotprod(d, d);

    printEnergies(system);

  } // end accept
  else {
    system.constants.iter_success =0;
    // reject for whole molecule
    for (int i=0; i<system.molecules[randm].atoms.size(); i++) {
      for (int n=0; n<3; n++) {
        system.molecules[randm].atoms[i].pos[n] = tmp_molecule.atoms[i].pos[n];
      }
    }
    system.molecules[randm].calc_center_of_mass();
  } // end reject
}
