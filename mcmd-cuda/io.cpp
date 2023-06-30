#include <iostream>
#include <string>
#include <strings.h>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <map>
#include <string>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <math.h>
#include <iomanip>
using namespace std;

/* xyz read-in of atoms */
void readInAtomsXYZ(System &system, string filename) {
  // FOR THIS WE ASSUME ALL ATOMS BELONG TO THE MOF (FROZEN ONLY).
  // check if it exists
  struct stat buffer;
  if (stat (filename.c_str(), &buffer) != 0) {
    std::cout << "ERROR: The input_atoms_xyz file " << filename.c_str() << " doesn't exist.";
    printf("\n");
    exit(EXIT_FAILURE);
  }

  string line;
  ifstream myfile (filename);
  if (myfile.is_open())
  {
    Molecule whatev;
    system.proto.push_back(whatev); // the first prototype (movable molecule type).
    Molecule current_molecule;
    system.molecules.push_back(whatev);
    int line_count=0;
    // loop through each line
    while ( getline (myfile,line) )
    {
      vector<string> myvector;
      istringstream iss(line);
      copy(
          istream_iterator<string>(iss),
          istream_iterator<string>(),
          back_inserter(myvector)
          );

      line_count++;
      if (line_count == 1 && myvector.size() > 1) {
        std::cout << "ERROR: The input_atoms_xyz file (" << filename.c_str() << ") is not a properly formatted .xyz file. The first line should be the total atom count only. Did you mean to use `input_atoms [.pdb file]`?";
        printf("\n");
        exit(EXIT_FAILURE);
      }
      if (line_count < 3) continue; // for .xyz, skip first 2 lines
      // skip blank lines
      if (myvector.size() != 0) {
        if (myvector[2] == "X" && myvector[3] == "BOX") continue; // skip box vertices

        //temporary class instance current_atom
        Atom current_atom;
        current_atom.name = myvector[0];
        // I have a database of defaults in classes.cpp
        current_atom.mass = system.constants.masses[convertElement(system,current_atom.name)];
        current_atom.eps = system.constants.eps[convertElement(system,current_atom.name)];
        current_atom.sig = system.constants.sigs[convertElement(system,current_atom.name)];
        current_atom.polar = system.constants.polars[convertElement(system,current_atom.name)];
        //==============================================================
        current_atom.V = 0.0;
        current_atom.PDBID = line_count - 2; // skipping first 2 lines
        current_atom.mol_name = "MOF";
        current_atom.frozen = 1;
        current_atom.mol_PDBID = 1;
        current_atom.pos[0] = stod(myvector[1]);
        current_atom.pos[1] = stod(myvector[2]);
        current_atom.pos[2] = stod(myvector[3]);
        if (myvector.size() > 4) current_atom.C = stod(myvector[4]) * system.constants.E2REDUCED;
        else current_atom.C = 0.; // default to zero charge unless the column is there
        system.molecules[0].atoms.push_back(current_atom);
        system.molecules[0].mass += current_atom.mass;
        system.constants.total_atoms++;  // add +1 to master atoms count
        system.stats.count_frozens++; // add +1 to frozen atoms count
      } // end if vector size nonzero
    }
    myfile.close();
    system.stats.count_frozen_molecules=1; // add +1 frozen molecule
    system.molecules[0].frozen = 1;
    system.molecules[0].PDBID = 1;
  }
  else {
    if (system.constants.sorbate_name.size() > 0) return;

    printf("ERROR: Unable to open %s. Exiting.\n",filename.c_str());
    std::exit(0);
  }

}

/* READ IN THE STARTING COORDINATES, CHARGES, MOLECULE ID'S FROM PDB */
void readInAtoms(System &system, string filename) {

  if (system.constants.readinxyz) {
    readInAtomsXYZ(system, filename);
  }
  else {

    // check if it exists
    struct stat buffer;
    if (stat (filename.c_str(), &buffer) != 0) {
      std::cout << "ERROR: The input_atoms file " << filename.c_str() << " doesn't exist.";
      printf("\n");
      exit(EXIT_FAILURE);
    }

    string line;
    ifstream myfile (filename);
    if (myfile.is_open())
    {
      int current_mol_id=-1; // Initializer. Will be changed.
      int mol_counter=-1;
      bool first_mover_passed = false;
      int first_mover_id = -1;
      int line_count=0;
      Molecule whatev;
      system.proto.push_back(whatev); // the first prototype.
      Molecule current_molecule; // initializer. Will be overwritten
      // loop through each line
      while ( getline (myfile,line) )
      {
        vector<string> myvector;
        istringstream iss(line);
        copy(
            istream_iterator<string>(iss),
            istream_iterator<string>(),
            back_inserter(myvector)
            );

        line_count++;
        if (line_count == 1 && myvector.size() < 2) {
          std::cout << "ERROR: The input_atoms file (" << filename.c_str() << ") is not a properly formatted .pdb file. All atom lines should contain ATOM in the first column. Did you mean to use `input_atoms_xyz [.xyz file]`?";
          printf("\n");
          exit(EXIT_FAILURE);

        }

        // skip blank lines
        if (myvector.size() != 0) {
          if (myvector[0] != "ATOM") continue; // skip anything in file that isn't an atom
          if (myvector[2] == "X" && myvector[3] == "BOX") continue; // skip box vertices

          //temporary class instance current_atom
          Atom current_atom;
          current_atom.name = myvector[2];
          // I have a database of defaults in classes.cpp
          // Those defaults will load unless the input column is there
          if (9 < myvector.size() && myvector[9] != "default") current_atom.mass = stod(myvector[9]);
          else current_atom.mass = system.constants.masses[convertElement(system,current_atom.name)];

          if (12 < myvector.size() && myvector[12] != "default") current_atom.eps = stod(myvector[12]);
          else current_atom.eps = system.constants.eps[convertElement(system,current_atom.name)];

          if (13 < myvector.size() && myvector[13] != "default") current_atom.sig = stod(myvector[13]);
          else current_atom.sig = system.constants.sigs[convertElement(system,current_atom.name)];

          if (11 < myvector.size() && myvector[11] != "default") current_atom.polar = stod(myvector[11]);
          else current_atom.polar = system.constants.polars[convertElement(system,current_atom.name)];

          // Tang Toennies params. If TT is active, LJ epsilon is used for B; LJ sigma for A...
          if (14 < myvector.size()) current_atom.c6 = stod(myvector[14]);
          if (15 < myvector.size()) current_atom.c8 = stod(myvector[15]);
          if (16 < myvector.size()) current_atom.c10 = stod(myvector[16]);


          current_atom.V = 0.0;
          current_atom.PDBID = stoi(myvector[1]); // pulled from input pdb column 2
          current_atom.mol_name = myvector[3];
          if (myvector[4] == "F")
            current_atom.frozen = 1;
          else
            current_atom.frozen = 0;
          // flag the first moving molecule as prototype sorbate
          // this will only return true once.
          if (myvector[4] == "M" && first_mover_passed == false) {
            first_mover_passed = true;
            first_mover_id = stoi(myvector[5]);
            system.proto[0].name = myvector[3];
            system.proto[0].frozen = 0;
            system.proto[0].PDBID = stoi(myvector[5]);
            system.proto[0].fugacity = system.constants.pres;
          }
          current_atom.mol_PDBID = stoi(myvector[5]);
          current_atom.pos[0] = stod(myvector[6]);
          current_atom.pos[1] = stod(myvector[7]);
          current_atom.pos[2] = stod(myvector[8]);
          current_atom.C = stod(myvector[10]) * system.constants.E2REDUCED;

          // create new molecule if needed.
          if (current_mol_id != stoi(myvector[5])) {
            // make a new molecule.
            mol_counter++;
            current_mol_id = stoi(myvector[5]);
            Molecule current_molecule;
            current_molecule.PDBID = current_mol_id;
            current_molecule.name = myvector[3];
            if (myvector[4] == "M")
              current_molecule.frozen = 0;
            else if (myvector[4] == "F")
              current_molecule.frozen = 1;
            system.molecules.push_back(current_molecule); // make the molecule

            if (myvector[4] == "M")
              system.stats.count_movables++;
            else if (myvector[4] == "F") {
              system.stats.count_frozen_molecules++; // add +1 frozen molecule
            }

          }

          // add atom to current molecule by default
          system.molecules[mol_counter].atoms.push_back(current_atom);
          system.molecules[mol_counter].mass += current_atom.mass;

          // and add current atom to prototype only if its in the first mover
          if (current_mol_id == first_mover_id) {
            system.proto[0].atoms.push_back(current_atom);
            system.proto[0].mass += current_atom.mass;
          }
          system.constants.total_atoms++;  // add +1 to master atoms count

          if (myvector[4] == "F")
            system.stats.count_frozens++; // add +1 to frozen atoms count
        } // end if vector size nonzero
      }
      myfile.close();


    }
    else {
      if (system.constants.sorbate_name.size() > 0) return;

      printf("ERROR: Unable to open %s. Exiting.\n",filename.c_str());
      std::exit(0);
    }
  } // end PDB readin
}


/* WRITE FRAME COORDINATE FOR TRAJECTORY FILE */
void writeXYZ(System &system, string filename, int frame, int step, double realtime, int mover_only_flag) {

  ofstream myfile;
  myfile.open (filename, ios_base::app);
  int totalatoms;

  if (mover_only_flag) totalatoms = system.constants.total_atoms - system.stats.count_frozens;
  else totalatoms = system.constants.total_atoms;

  if (system.constants.com_option) totalatoms++;

  myfile << to_string(totalatoms) + "\nFrame " + to_string(frame) + "; Step count: " + to_string(step) + "; Realtime (MD) = " + to_string(realtime) + "fs\n";

  for (int j = 0; j < system.molecules.size(); j++) {
    for (int i = 0; i < system.molecules[j].atoms.size(); i++) {
      if ((mover_only_flag && !system.molecules[j].atoms[i].frozen) || mover_only_flag==0) {
        myfile << system.molecules[j].atoms[i].name;
        myfile <<  "   ";
        // each coordinate has 9 decimal places, fill with zeros where needed.
        myfile << std::fixed << std::setprecision(9) << system.molecules[j].atoms[i].pos[0];
        myfile <<  "   ";
        myfile << std::setprecision(9) << system.molecules[j].atoms[i].pos[1];
        myfile <<  "   ";
        myfile << std::setprecision(9) << system.molecules[j].atoms[i].pos[2];
        myfile <<  "   ";
        myfile << std::setprecision(9) << system.molecules[j].atoms[i].C / system.constants.E2REDUCED; // write charge too
        myfile << "\n";
      }
    }
  }

  if (system.constants.com_option) {
    // get center of mass
    if (system.stats.count_movables > 0) {
      double* comfinal = centerOfMass(system);
      myfile << "COM   " << to_string(comfinal[0]) << "   " << to_string(comfinal[1]) << "   " << to_string(comfinal[2]) << "\n";
      delete[] comfinal;
    } else {
      myfile << "COM   " << "0.0" << "   " << "0.0" << "   " << "0.0" << "\n";
    }
  }

  if (!myfile.is_open()) {
    printf("Error opening XYZ restart file!\n");
    exit(1);
  }

  myfile.close();
}

/* WRITE PDB TRAJECTORY (TAKES restart.pdb and appends to trajectory file */
void writePDBtraj(System &system, string restartfile, string trajfile, int step) {
  std::ifstream ifile(restartfile.c_str(), std::ios::in);
  std::ofstream ofile(trajfile.c_str(), std::ios::out | std::ios::app);

  if (system.stats.count_movables < 1) return; // don't write empty trajectory
  ofile << "REMARK step=" << step << "\n";
  ofile << "REMARK total_molecules=" << system.molecules.size() << ", total_atoms=" << system.constants.total_atoms << "\n";
  ofile << "REMARK frozen_molecules=" << system.stats.count_frozen_molecules << ", movable_molecules=" << system.stats.count_movables << "\n";
  ofile << "REMARK frozen_atoms=" << system.stats.count_frozens << ", movable_atoms=" << (system.constants.total_atoms - system.stats.count_frozens) << "\n";

  if (!ifile.is_open()) {
    printf("Error opening PDB restart file! (in trajectory-writing function).\n");
    exit(1);
  } else {
    ofile << ifile.rdbuf(); // append contents of restartfile into trajfile
  }

  ofile << "ENDMDL\n";
}

/* WRITE PDB RESTART BACKUP FILE */
void writePDBrestartBak(System &system, string restartfile, string restartBakFile) {
  std::ifstream ifile(restartfile.c_str(), std::ios::in);
  std::ofstream ofile(restartBakFile.c_str(), std::ios::out);
  if (system.stats.count_movables < 1) return; // don't write empty trajectory

  if (!ifile.is_open()) {
    printf("Error opening PDB restart file! (in restart.pdb.bak-writing function).\n");
    exit(1);
  } else {
    ofile << ifile.rdbuf(); // append contents of restartfile into trajfile
  }

  ofile << "ENDMDL\n";
}


/* WRITE PDB RESTART FILE EVERY CORRTIME -- ONLY MOVABLES */
void writePDBmovables(System &system, string filename) {
  remove ( filename.c_str() );
  string frozenstring;
  FILE *f = fopen(filename.c_str(), "w");
  if (f == NULL)
  {
    printf("Error opening PDB movables restart file! (in movables restart-writing function).\n");
    exit(1);
  }
  for (int j=0; j<system.molecules.size(); j++) {
    for (int i=0; i<system.molecules[j].atoms.size(); i++) {
      if (system.molecules[j].atoms[i].frozen)
        continue; // skip frozens!
      else frozenstring = "M";
      if (!system.constants.pdb_long) {
        // this is default. VMD requires the "true" %8.3f
        fprintf(f, "ATOM  %5i %4s %3s %1s %3i    %8.3f%8.3f%8.3f %3.5f %3.5f %f %f %f\n",
            system.molecules[j].atoms[i].PDBID, // col 2
            system.molecules[j].atoms[i].name.c_str(), // 3
            system.molecules[j].atoms[i].mol_name.c_str(), // 4
            frozenstring.c_str(), // 5
            system.molecules[j].atoms[i].mol_PDBID, // 6
            system.molecules[j].atoms[i].pos[0], // 7
            system.molecules[j].atoms[i].pos[1],  // 8
            system.molecules[j].atoms[i].pos[2], //9
            system.molecules[j].atoms[i].mass, // 10
            system.molecules[j].atoms[i].C/system.constants.E2REDUCED,  // 11
            system.molecules[j].atoms[i].polar, // 12
            system.molecules[j].atoms[i].eps,  //13
            system.molecules[j].atoms[i].sig); //14
      }
      else if (system.constants.pdb_long) {
        fprintf(f, "ATOM  %5i %4s %3s %1s %3i %8.6f %8.6f %8.6f %3.6f %3.6f %f %f %f\n",
            system.molecules[j].atoms[i].PDBID, // col 2
            system.molecules[j].atoms[i].name.c_str(), // 3
            system.molecules[j].atoms[i].mol_name.c_str(), // 4
            frozenstring.c_str(), // 5
            system.molecules[j].atoms[i].mol_PDBID, // 6
            system.molecules[j].atoms[i].pos[0], // 7
            system.molecules[j].atoms[i].pos[1],  // 8
            system.molecules[j].atoms[i].pos[2], //9
            system.molecules[j].atoms[i].mass, // 10
            system.molecules[j].atoms[i].C/system.constants.E2REDUCED,  // 11
            system.molecules[j].atoms[i].polar, // 12
            system.molecules[j].atoms[i].eps,  //13
            system.molecules[j].atoms[i].sig); //14
      }
    } // end for atoms
  } // end for molecules

  // we don't need a box for this file 'cuz it should be in the MOF (frozen) file

  fclose(f);
}

/* WRITE PDB RESTART FILE AT STARTUP -- ONLY FROZENS */
void writePDBfrozens(System &system, string filename) {
  remove ( filename.c_str() );
  string frozenstring;
  FILE *f = fopen(filename.c_str(), "w");
  if (f == NULL)
  {
    printf("Error opening frozen PDB file.\n");
    exit(1);
  }
  for (int j=0; j<system.molecules.size(); j++) {
    for (int i=0; i<system.molecules[j].atoms.size(); i++) {
      if (!system.molecules[j].atoms[i].frozen)
        continue; // skip movables!
      else frozenstring = "F";
      if (!system.constants.pdb_long) {
        // this is default. VMD requires the "true" %8.3f
        fprintf(f, "ATOM  %5i %4s %3s %1s %3i    %8.3f%8.3f%8.3f %3.5f %3.5f %f %f %f\n",
            system.molecules[j].atoms[i].PDBID, // col 2
            system.molecules[j].atoms[i].name.c_str(), // 3
            system.molecules[j].atoms[i].mol_name.c_str(), // 4
            frozenstring.c_str(), // 5
            system.molecules[j].atoms[i].mol_PDBID, // 6
            system.molecules[j].atoms[i].pos[0], // 7
            system.molecules[j].atoms[i].pos[1],  // 8
            system.molecules[j].atoms[i].pos[2], //9
            system.molecules[j].atoms[i].mass, // 10
            system.molecules[j].atoms[i].C/system.constants.E2REDUCED,  // 11
            system.molecules[j].atoms[i].polar, // 12
            system.molecules[j].atoms[i].eps,  //13
            system.molecules[j].atoms[i].sig); //14
      }
      else if (system.constants.pdb_long) {
        fprintf(f, "ATOM  %5i %4s %3s %1s %3i %8.6f %8.6f %8.6f %3.6f %3.6f %f %f %f\n",
            system.molecules[j].atoms[i].PDBID, // col 2
            system.molecules[j].atoms[i].name.c_str(), // 3
            system.molecules[j].atoms[i].mol_name.c_str(), // 4
            frozenstring.c_str(), // 5
            system.molecules[j].atoms[i].mol_PDBID, // 6
            system.molecules[j].atoms[i].pos[0], // 7
            system.molecules[j].atoms[i].pos[1],  // 8
            system.molecules[j].atoms[i].pos[2], //9
            system.molecules[j].atoms[i].mass, // 10
            system.molecules[j].atoms[i].C/system.constants.E2REDUCED,  // 11
            system.molecules[j].atoms[i].polar, // 12
            system.molecules[j].atoms[i].eps,  //13
            system.molecules[j].atoms[i].sig); //14
      }
    } // end for atoms
  } // end for molecules

  // and draw the box if user desires
  if (system.constants.draw_box_option) {

    int i,j,k,p,q,diff,l,m,n;
    int box_labels[2][2][2];
    double box_occupancy[3];
    double box_pos[3];
    int last_mol_index = system.molecules.size() - 1;
    int last_mol_pdbid = system.molecules[last_mol_index].PDBID;
    int last_atom_pdbid = system.molecules[last_mol_index].atoms[system.molecules[last_mol_index].atoms.size() - 1].PDBID;
    int atom_box = last_atom_pdbid + 1;
    int molecule_box = last_mol_pdbid + 1;

    // draw the box points
    for(i = 0; i < 2; i++) {
      for(j = 0; j < 2; j++) {
        for(k = 0; k < 2; k++) {

          // make this frozen
          fprintf(f, "ATOM  ");
          fprintf(f, "%5d", atom_box);
          fprintf(f, " %-4.45s", "X");
          fprintf(f, " %-3.3s ", "BOX");
          fprintf(f, "%-1.1s", "F");
          fprintf(f, " %4d   ", molecule_box);

          // box coords
          box_occupancy[0] = ((double)i) - 0.5;
          box_occupancy[1] = ((double)j) - 0.5;
          box_occupancy[2] = ((double)k) - 0.5;


          for(p = 0; p < 3; p++)
            for(q = 0, box_pos[p] = 0; q < 3; q++)
              box_pos[p] += system.pbc.basis[q][p]*box_occupancy[q];

          for(p = 0; p < 3; p++)
            if(!system.constants.pdb_long)
              fprintf(f, "%8.3f", box_pos[p]);
            else
              fprintf(f, "%11.6f ", box_pos[p]);

          // null interactions
          fprintf(f, " %8.4f", 0.0);
          fprintf(f, " %8.4f", 0.0);
          fprintf(f, " %8.5f", 0.0);
          fprintf(f, " %8.5f", 0.0);
          fprintf(f, " %8.5f", 0.0);
          fprintf(f, "\n");

          box_labels[i][j][k] = atom_box;
          ++atom_box;

        } // for k
      } // for j
    } // for i

    // and draw the connecting lines
    for(i = 0; i < 2; i++) {
      for(j = 0; j < 2; j++) {
        for(k = 0; k < 2; k++) {

          for(l = 0; l < 2; l++) {
            for(m = 0; m < 2; m++) {
              for(n = 0; n < 2; n++) {

                diff = fabs(i - l) + fabs(j - m) + fabs(k - n);
                if(diff == 1)
                  fprintf(f, "CONECT %4d %4d\n", box_labels[i][j][k], box_labels[l][m][n]);

              } // n
            } // m
          } // l
        } // k
      } // j
    } // i

  } // if draw box is on
  // (end drawing the box)

  fclose(f);
}


/* WRITE PDB RESTART FILE EVERY CORRTIME */
void writePDB(System &system, string filename) {
  remove ( filename.c_str() );
  string frozenstring;
  FILE *f = fopen(filename.c_str(), "w");
  if (f == NULL)
  {
    printf("Error opening PDB restart file! (in restart-writing function).\n");
    exit(1);
  }
  for (int j=0; j<system.molecules.size(); j++) {
    for (int i=0; i<system.molecules[j].atoms.size(); i++) {
      if (system.molecules[j].atoms[i].frozen)
        frozenstring = "F";
      else if (!system.molecules[j].atoms[i].frozen)
        frozenstring = "M";
      if (!system.constants.pdb_long) {
        // this is default. VMD requires the "true" %8.3f
        fprintf(f, "ATOM  %5i %4s %3s %1s %3i    %8.3f%8.3f%8.3f %3.5f %3.5f %f %f %f\n",
            system.molecules[j].atoms[i].PDBID, // col 2
            system.molecules[j].atoms[i].name.c_str(), // 3
            system.molecules[j].atoms[i].mol_name.c_str(), // 4
            frozenstring.c_str(), // 5
            system.molecules[j].atoms[i].mol_PDBID, // 6
            system.molecules[j].atoms[i].pos[0], // 7
            system.molecules[j].atoms[i].pos[1],  // 8
            system.molecules[j].atoms[i].pos[2], //9
            system.molecules[j].atoms[i].mass, // 10
            system.molecules[j].atoms[i].C/system.constants.E2REDUCED,  // 11
            system.molecules[j].atoms[i].polar, // 12
            system.molecules[j].atoms[i].eps,  //13
            system.molecules[j].atoms[i].sig); //14
      }
      else if (system.constants.pdb_long) {
        fprintf(f, "ATOM  %5i %4s %3s %1s %3i %8.6f %8.6f %8.6f %3.6f %3.6f %f %f %f\n",
            system.molecules[j].atoms[i].PDBID, // col 2
            system.molecules[j].atoms[i].name.c_str(), // 3
            system.molecules[j].atoms[i].mol_name.c_str(), // 4
            frozenstring.c_str(), // 5
            system.molecules[j].atoms[i].mol_PDBID, // 6
            system.molecules[j].atoms[i].pos[0], // 7
            system.molecules[j].atoms[i].pos[1],  // 8
            system.molecules[j].atoms[i].pos[2], //9
            system.molecules[j].atoms[i].mass, // 10
            system.molecules[j].atoms[i].C/system.constants.E2REDUCED,  // 11
            system.molecules[j].atoms[i].polar, // 12
            system.molecules[j].atoms[i].eps,  //13
            system.molecules[j].atoms[i].sig); //14
      }
    } // end for atoms
  } // end for molecules



  // and draw the box if user desires
  if (system.constants.draw_box_option) {

    int i,j,k,p,q,diff,l,m,n;
    int box_labels[2][2][2];
    double box_occupancy[3];
    double box_pos[3];
    int last_mol_index = system.molecules.size() - 1;
    int last_mol_pdbid = system.molecules[last_mol_index].PDBID;
    int last_atom_pdbid = system.molecules[last_mol_index].atoms[system.molecules[last_mol_index].atoms.size() - 1].PDBID;
    int atom_box = last_atom_pdbid + 1;
    int molecule_box = last_mol_pdbid + 1;

    // draw the box points
    for(i = 0; i < 2; i++) {
      for(j = 0; j < 2; j++) {
        for(k = 0; k < 2; k++) {

          // make this frozen
          fprintf(f, "ATOM  ");
          fprintf(f, "%5d", atom_box);
          fprintf(f, " %-4.45s", "X");
          fprintf(f, " %-3.3s ", "BOX");
          fprintf(f, "%-1.1s", "F");
          fprintf(f, " %4d   ", molecule_box);

          // box coords
          box_occupancy[0] = ((double)i) - 0.5;
          box_occupancy[1] = ((double)j) - 0.5;
          box_occupancy[2] = ((double)k) - 0.5;


          for(p = 0; p < 3; p++)
            for(q = 0, box_pos[p] = 0; q < 3; q++)
              box_pos[p] += system.pbc.basis[q][p]*box_occupancy[q];

          for(p = 0; p < 3; p++)
            if(!system.constants.pdb_long)
              fprintf(f, "%8.3f", box_pos[p]);
            else
              fprintf(f, "%11.6f ", box_pos[p]);

          // null interactions
          fprintf(f, " %8.4f", 0.0);
          fprintf(f, " %8.4f", 0.0);
          fprintf(f, " %8.5f", 0.0);
          fprintf(f, " %8.5f", 0.0);
          fprintf(f, " %8.5f", 0.0);
          fprintf(f, "\n");

          box_labels[i][j][k] = atom_box;
          ++atom_box;

        } // for k
      } // for j
    } // for i

    // and draw the connecting lines
    for(i = 0; i < 2; i++) {
      for(j = 0; j < 2; j++) {
        for(k = 0; k < 2; k++) {

          for(l = 0; l < 2; l++) {
            for(m = 0; m < 2; m++) {
              for(n = 0; n < 2; n++) {

                diff = fabs(i - l) + fabs(j - m) + fabs(k - n);
                if(diff == 1)
                  fprintf(f, "CONECT %4d %4d\n", box_labels[i][j][k], box_labels[l][m][n]);

              } // n
            } // m
          } // l
        } // k
      } // j
    } // i

  } // if draw box is on
  // (end drawing the box in .pbd restart)

  fclose(f);
}

/* WRITE RUNNING ENERGY AVERAGE EVERY CORRTIME */
void writeThermo(System &system, double TE, double LKE, double RKE, double PE, double RD, double ES, double POL, double density, double temp, double pressure, int step, int N) {
  FILE *f = fopen(system.constants.thermo_output.c_str(), "a");
  if (f == NULL) {
    printf("Error opening thermo data file!\n");
    exit(1);
  }

  fprintf(f, "%i  %f  %f  %f  %f  %f  %f  %f %f %f %f %i\n",
      step, TE, LKE, RKE, PE, RD, ES, POL, density, temp, pressure, N);

  fclose(f);
}

void writeLAMMPSfiles(System &system) {

  printf("\n\nWriting LAMMPS input files...\n");

  // first write the input file
  FILE *f = fopen("lammps.in", "w");
  time_t rawtime;
  struct tm * timeinfo;
  char buffer[80];
  time (&rawtime);
  timeinfo = localtime(&rawtime);
  strftime(buffer,sizeof(buffer),"%m-%d-%Y at %H:%M:%S",timeinfo);
  std::string str(buffer);
  fprintf(f, "# LAMMPS input generated by MCMD on %s\n\n", str.c_str());

  fprintf(f, "# some variables for easy scripting later.\n");
  fprintf(f, "variable temperature equal %f\n", system.constants.temp);
  fprintf(f, "variable freq equal %i\n", (system.constants.mode == "mc")?system.constants.mc_corrtime : system.constants.md_corrtime);
  fprintf(f, "variable nstep equal %li\n", (system.constants.mode == "mc")?system.constants.finalstep : (int)ceil(system.constants.md_ft/system.constants.md_dt));

  fprintf(f, "\n# some global params\nunits real\nboundary p p p\natom_style full\n\n# the atoms file\nread_data lammps.data\n\n");
  // we need unique atoms list.
  vector<string> atomlabels;
  for (int i=0; i<system.molecules.size(); i++) {
    for (int j=0; j<system.molecules[i].atoms.size(); j++) {
      string theatom = system.molecules[i].atoms[j].name;
      if (std::find(atomlabels.begin(), atomlabels.end(), theatom) != atomlabels.end()) {
        // the atom is already in our unique set "atomlabels"
        continue;
      } else {
        atomlabels.push_back(theatom);
      }
    }
  }
  fprintf(f, "# site (pseudo-atom) masses\n");
  for (int x=0; x<atomlabels.size(); x++) {
    // find example of  atom type (for mass)
    double mass=0;
    for (int i=0; i<system.molecules.size(); i++) {
      for (int j=0; j<system.molecules[i].atoms.size(); j++) {
        if (system.molecules[i].atoms[j].name == atomlabels[x]) {
          mass = system.molecules[i].atoms[j].mass;// / system.constants.cM; // in amu
          // LAMMPS doesn't accept 0-mass atoms so we need to contrive this
          if (mass == 0) mass = 0.00001;
          break;
        }
      }
    }
    fprintf(f, "mass %i %.5f #%s\n", x+1, mass, atomlabels[x].c_str());
  }
  fprintf(f,"\n");

  fprintf(f,"timestep %f # femptoseconds (1e-15 s)\n", (system.constants.mode=="mc") ? 1.0 : system.constants.md_dt);


  fprintf(f, "\n# methods for potential/force calculations\n");
  fprintf(f, "kspace_style ewald/disp 1.0e-6\n");
  fprintf(f, "pair_style lj/cut/coul/long 2.5 %f # this gets overwritten by pair param's below\n\n# Atomic parameters (used for mixing)\n", system.pbc.cutoff);
  for (int x=0; x<atomlabels.size(); x++) {
    // find example of atom type (for params)
    double eps=0, sig=0, sigcut=0;
    for (int i=0; i<system.molecules.size(); i++) {
      for (int j=0; j<system.molecules[i].atoms.size(); j++) {
        if (system.molecules[i].atoms[j].name == atomlabels[x]) {
          eps = system.molecules[i].atoms[j].eps * system.constants.kbk; // to kcal/mol
          sig = system.molecules[i].atoms[j].sig; // * system.constants.mpmc2uff; // to A for LJ
          sigcut = sig * 2.5;
          break;
        }
      }
    }
    // write mixing rule parameters
    fprintf(f, "pair_coeff %i %i %.6f %.6f %.6f\n", x+1, x+1, eps, sig, sigcut);
  } // end unique atom (atom type) loop

  fprintf(f, "\nbond_style morse");
  fprintf(f, "\nangle_style fourier");
  fprintf(f, "\ndihedral_style fourier");
  fprintf(f, "\n#improper_style cvff\n");
  for (int i=0; i < (int)system.constants.uniqueBonds.size(); i++) {
    fprintf(f, "bond_coeff %i %f %f %f\n", i+1, system.constants.uniqueBonds[i].Dij, system.constants.uniqueBonds[i].alpha, system.constants.uniqueBonds[i].rij );
  }
  for (int it=0; it< (int)system.constants.uniqueAngles.size(); it++) {
    int i = system.constants.uniqueAngles[it].mol;
    int j = system.constants.uniqueAngles[it].atom1;
    int l = system.constants.uniqueAngles[it].atom2;
    int m = system.constants.uniqueAngles[it].atom3;
    double rij = system.constants.uniqueAngles[it].rij; // in Angstroms
    double rjk = system.constants.uniqueAngles[it].rjk;
    double t1 = system.constants.uniqueAngles[it].t1;
    double t2 = system.constants.uniqueAngles[it].t2;
    double t3 = system.constants.uniqueAngles[it].t3;
    double C2 = system.constants.uniqueAngles[it].C2; // 1/rad^2
    double C1 = system.constants.uniqueAngles[it].C1; // 1/rad
    double C0 = system.constants.uniqueAngles[it].C0; // 1
    double angle = get_angle(system, i, j, l, m);
    double rik = get_rik(system, rij, rjk, t3, angle);
    fprintf(f, "angle_coeff %i %f %f %f %f\n", it+1, get_Kijk(system, rik, t1, t2, t3), C0, C1, C2);
  }
  for (int it=0; it<(int)system.constants.uniqueDihedrals.size(); it++) {
    fprintf(f, "dihedral_coeff %i %i %f %f %f\n", it+1, 1, 0.5*system.constants.uniqueDihedrals[it].vjk, system.constants.uniqueDihedrals[it].n, system.constants.uniqueDihedrals[it].phi_ijkl);
  }


  fprintf(f, "\n# Apply LB mixing rules.\n");
  fprintf(f, "pair_modify mix arithmetic\n");

  fprintf(f, "\n# groups\n");
  fprintf(f, "group moving molecule > 1  # for a MOF simulation, the MOF is molecule 1\n");
  fprintf(f, "group frozen molecule 1\n");
  string showAtoms = "";
  string hideAtoms = "";
  for (int x=0; x<atomlabels.size(); x++) {
    bool found = false;
    // find example of  atom type (for mass)
    for (int i=0; i<system.molecules.size(); i++) {
      for (int j=0; j<system.molecules[i].atoms.size(); j++) {
        if (system.molecules[i].atoms[j].name == atomlabels[x]) {
          if (system.molecules[i].atoms[j].mass != 0) {
            showAtoms = showAtoms + " " + to_string(x+1);
            found = true;
            break;
          } else {
            hideAtoms = hideAtoms + " " + to_string(x+1);
            found = true;
            break;
          }
        }
      }
      if (found) break;
    } // end molecule i loop
  } // end unique atom names loop
  fprintf(f, "group show type %s\n", showAtoms.c_str());
  fprintf(f, "group hide type %s\n", hideAtoms.c_str());

  fprintf(f, "\n# exclusions\n");
  fprintf(f, "neigh_modify exclude molecule/intra frozen\n");
  fprintf(f, "neigh_modify exclude molecule/intra moving\n");

  fprintf(f, "# radial distribution");
  fprintf(f, "\n#compute rdf all rdf 50 [pairs... e.g. 8 1 8 8...]\n");

  fprintf(f, "\n# more variables etc. useful for MD");
  fprintf(f, "\nvariable step equal step\nvariable time equal step*2\nvariable timeNS equal time/1000000\nvariable diffusion_coeff equal c_themsd[4]/(6*time)*0.1 # cm^2/s\n");
  fprintf(f, "variable te equal c_pe+c_ke\n");
  fprintf(f, "compute pe all pe\ncompute ke all ke\ncompute themsd moving msd com yes average yes\ncompute movingtemp moving temp\nthermo_style custom step etotal ke pe evdwl ecoul\nthermo ${freq}\n");
  fprintf(f, "# dipoles computes\n");
  fprintf(f, "compute cc1 all chunk/atom molecule\n");
  fprintf(f, "compute dipoles all dipole/chunk cc1\n\n");

  string idSort = "";
  for (int x=0; x<atomlabels.size(); x++) {
    idSort = idSort + " " + atomlabels[x].c_str();
  }
  fprintf(f, "# write files (dumps)\ndump Dump all xyz ${freq} lammps_traj.xyz\ndump_modify Dump sort id\ndump_modify Dump element %s\n\n", idSort.c_str());

  fprintf(f, "# set NVT\n");
  fprintf(f, "velocity all create ${temperature} 12345 rot yes mom yes dist gaussian\n");
  fprintf(f, "fix rigid_nvt moving rigid/nvt molecule temp ${temperature} ${temperature} 100\n");
  fprintf(f, "fix 1 all ave/time 100 1 100 c_dipoles[*] file dipoles.out mode vector # to write the dipoles to file\n\n");
  fprintf(f, "\n# run\nrun ${nstep}");
  fclose(f);
  // DONE WITH LAMMPS INPUT FILE FOR MD SIMULATION. NOW WRITE THE .data FILE which contains
  // atoms, coords, box details, molecule assignment and charges

  FILE *f2 = fopen("lammps.data", "w");  //FILE*
  fprintf(f2, "%s\n\n", system.constants.jobname.c_str());
  fprintf(f2, "%i atoms\n", system.constants.total_atoms);
  fprintf(f2, "%i bonds\n", (int)system.constants.uniqueBonds.size());
  fprintf(f2, "%i angles\n", (int)system.constants.uniqueAngles.size());
  fprintf(f2, "%i dihedrals\n", (int)system.constants.uniqueDihedrals.size());
  fprintf(f2, "%i impropers\n\n", (int)system.constants.uniqueImpropers.size());

  fprintf(f2, "%i atom types\n", (int)atomlabels.size());
  fprintf(f2, "%i bond types\n", (int)system.constants.uniqueBonds.size());
  fprintf(f2, "%i angle types\n", (int)system.constants.uniqueAngles.size());
  fprintf(f2, "%i dihedral types\n", (int)system.constants.uniqueDihedrals.size());
  fprintf(f2, "%i improper types\n\n", (int)system.constants.uniqueImpropers.size());

  // we'll use the box vertices calculed in MCMD to get hi and lo params for the LAMMPS box.
  //double xlo=1e40, xhi=-1e40, ylo=1e40, yhi=-1e40, zlo=1e40, zhi=-1e40; // big numbers to start
  /*
     for (int v=0; v<8; v++) {
     if (system.pbc.box_vertices[v][0] < xlo) xlo = system.pbc.box_vertices[v][0];
     if (system.pbc.box_vertices[v][0] > xhi) xhi = system.pbc.box_vertices[v][0];
     if (system.pbc.box_vertices[v][1] < ylo) ylo = system.pbc.box_vertices[v][1];
     if (system.pbc.box_vertices[v][1] > yhi) yhi = system.pbc.box_vertices[v][1];
     if (system.pbc.box_vertices[v][2] < zlo) zlo = system.pbc.box_vertices[v][2];
     if (system.pbc.box_vertices[v][2] > zhi) zhi = system.pbc.box_vertices[v][2];
     }*/
  double xlo = -system.pbc.x_length/2.;
  double xhi = system.pbc.x_length/2.;
  double ylo = -system.pbc.y_length/2.;
  double yhi = system.pbc.y_length/2.;
  double zlo = -system.pbc.z_length/2.;
  double zhi = system.pbc.z_length/2.;

  //double a=system.pbc.a;
  double b=system.pbc.b;
  double c=system.pbc.c;
  double alpha=system.pbc.alpha*M_PI/180.;
  double beta=system.pbc.beta*M_PI/180.;
  double gamma=system.pbc.gamma*M_PI/180.;

  double xy = b*cos(gamma);
  double xz = c*cos(beta);
  double ly = sqrt(b*b - xy*xy);
  double yz = (b*c*cos(alpha) - xy*xz)/ly;
  double lz = sqrt(c*c - xz*xz - yz*yz);

  if (fabs(xy) < 1e-6) xy=0;
  if (fabs(xz) < 1e-6) xz=0;
  if (fabs(yz) < 1e-6) yz=0;

  fprintf(f2, "%.6f %.6f xlo xhi\n",xlo,xhi);
  fprintf(f2, "%.6f %.6f ylo yhi\n",ylo,yhi);
  fprintf(f2, "%.6f %.6f zlo zhi\n",zlo,zhi);
  fprintf(f2, "%.6f %.6f %.6f xy xz yz\n\n", xy, xz, yz);

  fprintf(f2, "Atoms\n\n");
  // now the atom list in LAMMPS style input
  int counter=1;
  int atomtype;
  for (int i=0; i<system.molecules.size(); i++) {
    for (int j=0; j<system.molecules[i].atoms.size(); j++) {
      for (int x=0; x<atomlabels.size(); x++) {
        if (atomlabels[x] == system.molecules[i].atoms[j].name) {
          atomtype = x+1;
          fprintf(f2, "%i %i %i %.5f %.5f %.5f %.5f #%s\n",
              counter, system.molecules[i].PDBID, atomtype,
              system.molecules[i].atoms[j].C / system.constants.E2REDUCED,
              system.molecules[i].atoms[j].pos[0],
              system.molecules[i].atoms[j].pos[1],
              system.molecules[i].atoms[j].pos[2],
              atomlabels[x].c_str());
          counter++;
          break;
        }
      } // end atom labels
    } // end j atoms in mol
  } // end i molecules loop

  if (system.constants.uniqueBonds.size() > 0) {
    fprintf(f2, "\nBonds\n\n");
    for (int z=0; z< system.constants.uniqueBonds.size(); z++) {
      int i = system.constants.uniqueBonds[z].atom1 + 1;
      int j = system.constants.uniqueBonds[z].atom2 + 1;
      fprintf(f2, "%i %i %i %i\n", z+1, z+1, i, j);
    }
  }

  if (system.constants.uniqueAngles.size() > 0) {
    fprintf(f2, "\nAngles\n\n");
    for (int z=0; z<system.constants.uniqueAngles.size(); z++) {
      int i = system.constants.uniqueAngles[z].atom1 + 1;
      int j = system.constants.uniqueAngles[z].atom2 + 1;
      int k = system.constants.uniqueAngles[z].atom3 + 1;
      fprintf(f2, "%i %i %i %i %i\n", z+1, z+1, i, j, k );
    }
  }

  if (system.constants.uniqueDihedrals.size() > 0) {
    fprintf(f2, "\nDihedrals\n\n");
    for (int z=0; z<system.constants.uniqueDihedrals.size(); z++) {
      int i = system.constants.uniqueDihedrals[z].atom1 + 1;
      int j = system.constants.uniqueDihedrals[z].atom2 + 1;
      int k = system.constants.uniqueDihedrals[z].atom3 + 1;
      int l = system.constants.uniqueDihedrals[z].atom4 + 1;
      fprintf(f2, "%i %i %i %i %i %i\n", z+1, z+1, i,j,k,l);
    }
  }

  if (system.constants.uniqueImpropers.size() > 0) {
    fprintf(f2, "\nImpropers\n\n");
    for (int z=0; z<system.constants.uniqueImpropers.size(); z++) {
      int i = system.constants.uniqueImpropers[z].atom1 + 1;
      int j = system.constants.uniqueImpropers[z].atom2 + 1;
      int k = system.constants.uniqueImpropers[z].atom3 + 1;
      int l = system.constants.uniqueImpropers[z].atom4 + 1;
      fprintf(f2, "%i %i %i %i %i %i\n", z+1, z+1, i,j,k,l);
    }
  }

  fclose(f2);

  printf("Done generating LAMMPS inputs from loaded data in MCMD.\n");
  printf(" --> Written to [[ lammps.in ]] and [[ lammps.data ]]\n\n");
}

/* READ INPUT FILE PARAMETERS AND OPTIONS */
void readInput(System &system, char* filename) {
  // check if it exists
  struct stat buffer;
  if (stat (filename, &buffer) != 0) {
    std::cout << "ERROR: The MCMD input file " << filename << " doesn't exist.";
    printf("\n");
    exit(EXIT_FAILURE);
  }

  printf("Reading input parameters from %s.\n",filename);

  string line;
  ifstream myfile (filename);
  if (!myfile.is_open()) {
    std::cout << "ERROR: failed to open input file " << filename << ".";
    printf("\n");
    exit(EXIT_FAILURE);
  }

  while ( getline (myfile,line) )
  {
    vector<string> lc;
    istringstream iss(line);
    copy(
        istream_iterator<string>(iss),
        istream_iterator<string>(),
        back_inserter(lc)
        );

    if (lc.empty()) continue;  // ignore blank lines

    if (!strncasecmp(lc[0].c_str(), "!", 1) || (!strncasecmp(lc[0].c_str(), "#", 1)))
      continue; // treat ! and # as comments

    if (!strcasecmp(lc[0].c_str(),"name")) {
      system.constants.jobname = lc[1].c_str();
      std::cout << "Got job name = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(),"restart")) {
      system.constants.restart_mode = 1;
      std::cout << "Got restart option. Restarting previous job from this directory.";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "mode")) {
      std::transform(lc[1].begin(), lc[1].end(), lc[1].begin(), ::tolower);
      system.constants.mode = lc[1].c_str();
      std::cout << "Got mode = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "gpu")) {
      if (!strcasecmp(lc[1].c_str(),"on"))
        system.constants.gpu = 1;
      std::cout << "Got GPU option = " << lc[1].c_str();
      printf("\n");
    } else if (!strcasecmp(lc[0].c_str(), "device_block_size")) {
      system.constants.device_block_size = atoi(lc[1].c_str());
      std::cout << "Got device block size (threads per block) = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "charge")) {
      system.constants.user_charge = atoi(lc[1].c_str());
      std::cout << "Got user-input charge = " << lc[1].c_str() << " e";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "ensemble")) {
      if (!strcasecmp(lc[1].c_str(),"nvt")) {
        system.constants.ensemble = ENSEMBLE_NVT;
        system.constants.ensemble_str = "NVT";
      }
      else if (!strcasecmp(lc[1].c_str(),"nve")) {
        system.constants.ensemble = ENSEMBLE_NVE;
        system.constants.ensemble_str = "NVE";
      }
      else if (!strcasecmp(lc[1].c_str(),"uvt")) {
        system.constants.ensemble = ENSEMBLE_UVT;
        system.constants.ensemble_str = "uVT";
      }
      else if (!strcasecmp(lc[1].c_str(),"npt")) {
        system.constants.ensemble = ENSEMBLE_NPT;
        system.constants.ensemble_str = "NPT";
      }
      std::cout << "Got ensemble = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "bias_uptake") || !strcasecmp(lc[0].c_str(), "uptake_bias")) {
      system.constants.bias_uptake = atof(lc[1].c_str());
      if (lc.size() > 2) {
        std::transform(lc[2].begin(), lc[2].end(), lc[2].begin(), ::tolower);
        system.constants.bias_uptake_unit = lc[2];
        system.constants.bias_uptake_switcher=1;
        std::cout << "Got uptake bias = " << lc[1].c_str() << " " << lc[2].c_str();
        printf("\n");
      } else {
        std::cout << "ERROR: You used the bias_uptake option but did not specify a unit. Use, e.g. `bias_uptake 2.5 wt%`.";
        printf("\n");
        exit(EXIT_FAILURE);
      }
    } else if (!strcasecmp(lc[0].c_str(), "sorbate_name")) {
      std::transform(lc[1].begin(), lc[1].end(), lc[1].begin(), ::tolower);
      system.constants.sorbate_name.push_back(lc[1].c_str());
      std::cout << "Got sorbate model name 1 = " << lc[1].c_str();
      printf("\n");

      for (int i=2; i<=10; i++) { // so max sorbates is 10.
        if (lc.size() >= (i+1)) {
          std::transform(lc[i].begin(), lc[i].end(), lc[i].begin(), ::tolower);
          system.constants.sorbate_name.push_back(lc[i].c_str());
          std::cout << "Got sorbate model name " << i << " = " << lc[i].c_str();
          printf("\n");
        }
      }

    } else if (!strcasecmp(lc[0].c_str(), "sorbate_fugacities") || (!strcasecmp(lc[0].c_str(), "user_fugacities")) ) {
      system.constants.sorbate_fugacity.push_back(atof(lc[1].c_str()));
      std::cout << "Got fugacity for sorbate 1 = " << lc[1].c_str();
      printf("\n");

      for (int i=2; i<=10; i++) {
        if (lc.size() >= (i+1)) {
          system.constants.sorbate_fugacity.push_back(atof(lc[i].c_str()));
          std::cout << "Got fugacity for sorbate " << i << " = " << lc[i].c_str();
          printf("\n");
        }
      }

    } else if (!strcasecmp(lc[0].c_str(), "sorbate_dofs")) {
      system.constants.sorbate_dof.push_back(atof(lc[1].c_str()));
      std::cout << "Got degrees of freedom for sorbate 1 = " << lc[1].c_str();
      printf("\n");

      for (int i=2; i<=10; i++) {
        if (lc.size() >= (i+1)) {
          system.constants.sorbate_dof.push_back(atof(lc[i].c_str()));
          std::cout << "Got degrees of freedom for sorbate " << i << " = " << lc[i].c_str();
          printf("\n");
        }
      }

    } else if (!strcasecmp(lc[0].c_str(), "fugacity_single") || !strcasecmp(lc[0].c_str(), "fugacity")) {
      system.constants.fugacity_single = 1;
      std::transform(lc[1].begin(), lc[1].end(), lc[1].begin(), ::tolower);
      if (lc[1] == "none") lc[1] = "off"; // allow "none" for "off"
      system.constants.fugacity_single_sorbate = lc[1];
      if (lc[1] != "h2" && lc[1] != "co2" && lc[1] != "ch4" && lc[1] != "n2" && lc[1] != "off") {
        std::cout << "ERROR: fugacity_single input not recognized. Available options are h2, co2, ch4, and n2.";
        std::exit(0);
      } else {
        std::cout << "Got fugacity_single sorbate selection = " << lc[1].c_str();
        printf("\n");
      }
    } else if (!strcasecmp(lc[0].c_str(), "co2_fit_fugacity")) {
      if (!strcasecmp(lc[1].c_str(), "on")) {
        system.constants.co2_fit_fugacity = 1;
      }
      std::cout << "Got co2_fit_fugacity = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "input_atoms_xyz")) {
      if (lc.size() > 1) {
        system.constants.readinxyz = 1;
        system.constants.atom_file = lc[1].c_str();
      }

      if (system.constants.readinxyz == 1)
        std::cout << "Got xyz input file = " << lc[1].c_str();
      printf("\n");

      // BASIS STUFF.
      // If user inputs x_length, y_length, z_length, assume 90deg. angles
    } else if (!strcasecmp(lc[0].c_str(), "x_length")) {
      system.pbc.x_length = atof(lc[1].c_str());
      system.pbc.basis[0][0] = atof(lc[1].c_str());
      system.pbc.basis[0][1] = 0;
      system.pbc.basis[0][2] = 0;
      std::cout << "Got x_length = " << lc[1].c_str() << " A";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "y_length")) {
      system.pbc.y_length = atof(lc[1].c_str());
      system.pbc.basis[1][1] = atof(lc[1].c_str());
      system.pbc.basis[1][0] = 0;
      system.pbc.basis[1][2] = 0;
      std::cout << "Got y_length = " << lc[1].c_str() << " A";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "z_length")) {
      system.pbc.z_length = atof(lc[1].c_str());
      system.pbc.basis[2][2] = atof(lc[1].c_str());
      system.pbc.basis[2][0] = 0;
      system.pbc.basis[2][1] = 0;
      std::cout << "Got z_length = " << lc[1].c_str() << " A";
      printf("\n");

      system.pbc.calcCarBasis();

      // OR EXACT BASIS INPUT (by vectors)
    } else if (!strcasecmp(lc[0].c_str(), "basis1") || !strcasecmp(lc[0].c_str(), "a")) {
      for (int n=0; n<3; n++)
        system.pbc.basis[0][n] = atof(lc[n+1].c_str());
      system.pbc.x_length = system.pbc.basis[0][0];

      std::cout << "Got basis1 = " << lc[1].c_str() << " " << lc[2].c_str() << " " << lc[3].c_str();
      printf("\n");
    } else if (!strcasecmp(lc[0].c_str(), "basis2") || !strcasecmp(lc[0].c_str(), "b")) {
      for (int n=0; n<3; n++)
        system.pbc.basis[1][n] = atof(lc[n+1].c_str());
      system.pbc.y_length = system.pbc.basis[1][1];

      std:: cout << "Got basis2 = " << lc[1].c_str() << " " << lc[2].c_str() << " " << lc[3].c_str();
      printf("\n");
    } else if (!strcasecmp(lc[0].c_str(), "basis3") || !strcasecmp(lc[0].c_str(), "c")) {
      for (int n=0; n<3; n++)
        system.pbc.basis[2][n] = atof(lc[n+1].c_str());
      system.pbc.z_length = system.pbc.basis[2][2];

      std:: cout << "Got basis3 = " << lc[1].c_str() << " " << lc[2].c_str() << " " << lc[3].c_str();
      printf("\n");

      system.pbc.calcCarBasis();

    } else if (!strcasecmp(lc[0].c_str(), "carbasis")) {
      double a = atof(lc[1].c_str());
      double b = atof(lc[2].c_str());
      double c = atof(lc[3].c_str());
      double alpha = atof(lc[4].c_str());
      double beta = atof(lc[5].c_str());
      double gamma = atof(lc[6].c_str());

      system.pbc.a = a;
      system.pbc.b = b;
      system.pbc.c = c;
      system.pbc.alpha = alpha;
      system.pbc.beta = beta;
      system.pbc.gamma = gamma;
      system.pbc.x_length = a;
      system.pbc.y_length = b;
      system.pbc.z_length = c;

      system.pbc.calcNormalBasis();

      std::cout << "Got .car basis: a,b,c = " << lc[1].c_str() << ", " << lc[2].c_str() << ", " << lc[3].c_str();
      printf("\n");
      std::cout << "Got .car basis alpha,beta,gamma = " << lc[4].c_str() << ", " << lc[5].c_str() << ", " << lc[6].c_str();
      printf("\n");
    } else if (!strcasecmp(lc[0].c_str(), "integrator")) {
      if (!strcasecmp(lc[1].c_str(), "rk4"))
        system.constants.integrator = INTEGRATOR_RK4;
      else if (!strcasecmp(lc[1].c_str(), "vv"))
        system.constants.integrator = INTEGRATOR_VV;

      std::cout << "Got integrator (for MD simulation) = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "feynman_hibbs") || !strcasecmp(lc[0].c_str(), "fh") || !strcasecmp(lc[0].c_str(), "feynmann_hibbs")) {  // allow for typo of R.F.'s name
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.feynman_hibbs = 1;
      std::cout << "Got Feynman-Hibbs correction option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "feynman_hibbs_order") || !strcasecmp(lc[0].c_str(), "fh_order") || !strcasecmp(lc[0].c_str(), "feynmann_hibbs_order")) { // allow typo,lol
      system.constants.fh_order = atoi(lc[1].c_str());
      std::cout << "Got Feynman-Hibbs order = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "input_atoms")) {
      system.constants.atom_file = lc[1].c_str();
      std::cout << "Got input atoms file name = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "restart_pdb")) {
      system.constants.restart_pdb = lc[1].c_str();
      std::cout << "Got restart output file = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "thermo_output")) {
      system.constants.thermo_output = lc[1].c_str();
      std::cout << "Got thermo output file = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "stepsize")) {
      system.constants.stepsize = atoi(lc[1].c_str());
      std::cout << "Got step size = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "finalstep") || !strcasecmp(lc[0].c_str(), "steps") || !strcasecmp(lc[0].c_str(), "numsteps")) {
      system.constants.finalstep = atol(lc[1].c_str());
      std::cout << "Got total steps = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "dist_within")) {
      if (!strcasecmp(lc[1].c_str(),"on"))
        system.constants.dist_within_option = 1;
      else system.constants.dist_within_option = 0;
      std::cout << "Got dist_within option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "dist_within_target")) {
      system.constants.dist_within_target = lc[1].c_str();
      std::cout << "Got dist_within_target atom = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "dist_within_radius")) {
      system.constants.dist_within_radius = atof(lc[1].c_str());
      std::cout << "Got dist_within_radius = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "full_A_matrix_option") || !strcasecmp(lc[0].c_str(), "full_A_matrix")) {
      if (!strcasecmp(lc[1].c_str(),"on")) {
        system.constants.full_A_matrix_option=1;
      }
      else if (!strcasecmp(lc[1].c_str(),"off")) {
        system.constants.full_A_matrix_option=0;
      }
      std::cout << "Got full polarization A matrix option = " << lc[1].c_str();
      printf("\n");
    } else if (!strcasecmp(lc[0].c_str(), "auto_reject_option") || !strcasecmp(lc[0].c_str(), "auto_reject")) {
      if (!strcasecmp(lc[1].c_str(),"off")) system.constants.auto_reject_option=0;
      std::cout << "Got auto_reject_option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "auto_reject_r")) {
      system.constants.auto_reject_r = atof(lc[1].c_str());
      std::cout << "Got auto-reject distance = " << lc[1].c_str() << " Angstroms.";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "auto_center")) {
      if (!strcasecmp(lc[1].c_str(),"on"))
        system.constants.autocenter = 1;
      else {
        system.constants.ac_x = 0;
        system.constants.ac_y = 0;
        system.constants.ac_z = 0;
        if (!strcasecmp(lc[1].c_str(),"x")) {
          system.constants.ac_x = 1;
        } else if (!strcasecmp(lc[1].c_str(), "y")) {
          system.constants.ac_y = 1;
        } else if (!strcasecmp(lc[1].c_str(), "z")) {
          system.constants.ac_z = 1;
        } else if (!strcasecmp(lc[1].c_str(), "xy")) {
          system.constants.ac_x=1;
          system.constants.ac_y=1;
        } else if (!strcasecmp(lc[1].c_str(), "xz")) {
          system.constants.ac_x=1;
          system.constants.ac_z=1;
        } else if (!strcasecmp(lc[1].c_str(), "yz")) {
          system.constants.ac_y=1;
          system.constants.ac_z=1;
        } else if (!strcasecmp(lc[1].c_str(), "xyz")) {
          system.constants.ac_x=1;
          system.constants.ac_y=1;
          system.constants.ac_z=1;
        } else {
          system.constants.autocenter = 0;
        }
      }
      std::cout << "Got auto-center-atoms-to-origin option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "no_zero_option")) {
      if (!strcasecmp(lc[1].c_str(),"on"))
        system.constants.no_zero_option = 1;
      std::cout << "Got no-zero-molecules-option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "md_corrtime")) {
      system.constants.md_corrtime = atoi(lc[1].c_str());
      std::cout << "Got MD corrtime = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "md_init_vel")) {
      system.constants.md_manual_init_vel = 1;
      system.constants.md_init_vel = atof(lc[1].c_str());
      std::cout << "Got MD initial velocity for all molecules = " << lc[1].c_str() << " A/fs.";
      printf("\n");
    } else if (!strcasecmp(lc[0].c_str(), "md_mode")) {
      if (lc[1] == "atomic")
        system.constants.md_mode = MD_ATOMIC;
      else if (lc[1] == "molecular" || lc[1] == "rigid")
        system.constants.md_mode = MD_MOLECULAR;
      else if (lc[1] == "flexible") {
        system.constants.md_mode = MD_FLEXIBLE;
      }
      std::cout << "Got MD mode = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "md_pbc")) {
      if (!strcasecmp(lc[1].c_str(),"on")) {
        system.constants.md_pbc = 1;
        system.constants.all_pbc =1;
      }
      else {
        system.constants.md_pbc = 0;
        system.constants.all_pbc = 0;
      }
      std::cout << "Got MD PBC option = " << lc[1].c_str();
      printf("\n");
      if (system.constants.md_pbc == 0) {
        printf("ERROR: GPU kernel is not available when MD PBC is off\n"); 
        exit(EXIT_FAILURE);
      }

    } else if (!strcasecmp(lc[0].c_str(), "mc_pbc")) {
      if (!strcasecmp(lc[1].c_str(),"on")) {
        system.constants.mc_pbc = 1;
        system.constants.all_pbc = 1;
      }
      else {
        system.constants.mc_pbc = 0;
        system.constants.all_pbc = 0;
      }
      std::cout << "Got MC PBC option = " << lc[1].c_str();
      printf("\n");
      if (system.constants.mc_pbc == 0) {
        system.constants.ewald_es = 0;
        system.constants.rd_lrc = 0;
        system.constants.polar_pbc = 0;
        system.constants.all_pbc = 0;
      }
    } else if (!strcasecmp(lc[0].c_str(), "omp") || !strcasecmp(lc[0].c_str(), "openmp")) {
      system.constants.openmp_threads = atoi(lc[1].c_str());
      std::cout << "OpenMP activated. Threads requested = " << lc[1].c_str();
      printf("\n");
    } else if (!strcasecmp(lc[0].c_str(), "external_force")) {
      if ((int)lc.size() < 5) {
        printf("ERROR: The external_force command needs the following syntax:\nexternal_force [x] [y] [z] [step_count] e.g.   0.2 0 0 5.\n");
        exit(EXIT_FAILURE);
      }
      system.constants.md_external_force = 1;
      for (int n=0; n<3; n++)
        system.constants.external_force_vector[n] = atof(lc[n+1].c_str())/system.constants.kb/1e10/1e9; // convert from nN (semi-intuitive force)   to   K/A (MCMD force)
      system.constants.md_external_force_freq = atoi(lc[4].c_str()); // apply ext. force every x steps
      std::cout << "Got external force vector = [ " << lc[1].c_str() << ", " << lc[2].c_str() << ", " << lc[3].c_str() << " ] nanoNewtons every " << lc[4].c_str() << " steps.";
      printf("\n");
      printf("So the ext. force in K/A is [ %.3f %.3f %.3f ].\n",
          system.constants.external_force_vector[0],
          system.constants.external_force_vector[1],
          system.constants.external_force_vector[2]);

      if (!(system.constants.md_external_force_freq > 0)) {
        printf("ERROR: The input value for external force step-frequency ( %s ) is not valid input. Must be an integer.\n",lc[4].c_str());
        exit(EXIT_FAILURE);
      }

    } else if (!strcasecmp(lc[0].c_str(), "simulated_annealing")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.simulated_annealing = 1;
      else system.constants.simulated_annealing = 0;
      std::cout << "Got simulated annealing option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "simulated_annealing_target")) {
      system.constants.sa_target = atof(lc[1].c_str());
      std::cout << "Got simulated annealing target temperature = " << lc[1].c_str() << " K";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "simulated_annealing_schedule")) {
      system.constants.sa_schedule = atof(lc[1].c_str());
      std::cout << "Got simulated annealing schedule = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "md_insert_frequency")) {
      system.constants.md_insert_attempt = atoi(lc[1].c_str());
      std::cout << "Got MD uVT insert/delete attempt frequency = every " << lc[1].c_str() << " timesteps.";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "step_offset")) {
      system.constants.step_offset = atoi(lc[1].c_str());
      std::cout << "Got step offset = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "draw_box_option")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.draw_box_option = 1;
      else system.constants.draw_box_option = 0;
      std::cout << "Got draw-box-option for PDB output = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "histogram")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.histogram_option = 1;
      else system.constants.histogram_option = 0;
      std::cout << "Got histogram option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "histogram_output")) {
      system.constants.output_histogram = lc[1].c_str();
      std::cout << "Got histogram output file = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "histogram_resolution")) {
      system.hist_resolution = atof(lc[1].c_str());
      std::cout << "Got histogram resolution = " << lc[1].c_str() << " A";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "mc_corrtime")) {
      system.constants.mc_corrtime = atoi(lc[1].c_str());
      std::cout << "Got MC corrtime = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "temperature")) {
      system.constants.temp = atof(lc[1].c_str());
      std::cout << "Got temperature = " << lc[1].c_str() << " K";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "pressure")) {
      system.constants.pres = atof(lc[1].c_str());
      std::cout << "Got pressure = " << lc[1].c_str() << " atm";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "insert_factor") || !strcasecmp(lc[0].c_str(), "insert_probability")) {
      system.constants.insert_factor = atof(lc[1].c_str());
      std::cout << "Got insert/delete factor = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "volume_change")) {
      system.constants.volume_change = atof(lc[1].c_str());
      std::cout << "Got volume change factor = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "rotate_option")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.rotate_option = 1;
      else system.constants.rotate_option = 0;
      std::cout << "Got rotate option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "rotate_angle_factor")) {
      system.constants.rotate_angle_factor = atof(lc[1].c_str());
      std::cout << "Got rotate angle factor = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "output_traj_xyz")) {
      system.constants.output_traj = lc[1].c_str();
      std::cout << "Got output trajectory XYZ filename = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "output_traj_pdb")) {
      system.constants.output_traj_pdb = lc[1].c_str();
      std::cout << "Got output trajectory PDB filename = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "xyz_traj_option")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.xyz_traj_option = 1;
      else system.constants.xyz_traj_option = 0;
      std::cout << "Got XYZ trajectory output option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "pdb_traj_option")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.pdb_traj_option = 1;
      std::cout << "Got PDB trajectory output option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "vcp_factor")) {
      system.constants.vcp_factor = atof(lc[1].c_str());
      std::cout << "Got volume change probability factor = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "displace_factor")) {
      system.constants.displace_factor = atof(lc[1].c_str());
      std::cout << "Got displace factor = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "big_pdb_traj")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.pdb_bigtraj_option = 1;
      std::cout << "Got Big PDB trajectory option (includes frozens every step) = " << lc[1].c_str();
      printf("\n");
    } else if (!strcasecmp(lc[0].c_str(), "big_xyz_traj")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.xyz_traj_movers_option=0;
      std::cout << "Got Big XYZ trajectory option (includes frozens every step) = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "potential_form")) {
      std::transform(lc[1].begin(), lc[1].end(), lc[1].begin(), ::tolower);
      if (!strcasecmp(lc[1].c_str(),"lj"))
        system.constants.potential_form = POTENTIAL_LJ;
      else if (!strcasecmp(lc[1].c_str(),"ljes"))
        system.constants.potential_form = POTENTIAL_LJES;
      else if (!strcasecmp(lc[1].c_str(),"ljpolar"))
        system.constants.potential_form = POTENTIAL_LJPOLAR;
      else if (!strcasecmp(lc[1].c_str(),"ljespolar"))
        system.constants.potential_form = POTENTIAL_LJESPOLAR;
      else if (!strcasecmp(lc[1].c_str(),"commy"))
        system.constants.potential_form = POTENTIAL_COMMY;
      else if (!strcasecmp(lc[1].c_str(),"commyes"))
        system.constants.potential_form = POTENTIAL_COMMYES;
      else if (!strcasecmp(lc[1].c_str(),"commyespolar"))
        system.constants.potential_form = POTENTIAL_COMMYESPOLAR;
      else if (!strcasecmp(lc[1].c_str(),"tt"))
        system.constants.potential_form = POTENTIAL_TT;
      else if (!strcasecmp(lc[1].c_str(),"ttes"))
        system.constants.potential_form = POTENTIAL_TTES;
      else if (!strcasecmp(lc[1].c_str(),"ttespolar"))
        system.constants.potential_form = POTENTIAL_TTESPOLAR;

      std::cout << "Got potential form = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "polar_max_iter")) {
      system.constants.polar_max_iter = atoi(lc[1].c_str());
      std::cout << "Got polarization iterations = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "polar_palmo")) {
      if (!strcasecmp(lc[1].c_str(),"off")) system.constants.polar_palmo = 0;
      std::cout << "Got Palmo Polarization = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "polar_precision") || !strcasecmp(lc[0].c_str(), "dipole_precision")) {
      system.constants.polar_precision = atof(lc[1].c_str());
      std::cout << "Got required dipole precision (for iterative routine) = " << lc[1].c_str() << " Debye.";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "com_option")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.com_option = 1;
      else system.constants.com_option = 0;
      std::cout << "Got center-of-mass option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "free_volume")) {
      system.constants.free_volume = atof(lc[1].c_str());
      std::cout << "Got free volume = " << lc[1].c_str() << " A^3.";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "write_lammps") || !strcasecmp(lc[0].c_str(), "lammps_write")) {
      if (!strcasecmp(lc[1].c_str(),"on")) {
        system.constants.write_lammps = 1;
      }
      std::cout << "Got LAMMPS input file write option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "md_dt")) {

      system.constants.md_dt = atof(lc[1].c_str());
      std::cout << "Got MD timestep = " << system.constants.md_dt << " fs";
      printf("\n");

      system.constants.md_thermostat_probab = system.constants.md_thermostat_freq *
        exp(-system.constants.md_thermostat_freq * system.constants.md_dt);
      std::cout << "The MD thermostat heat-bath collision probability = " << system.constants.md_thermostat_probab;
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "md_ft")) {

      // default fs

      if (lc.size() < 3 || lc[2] == "!") system.constants.md_ft = atof(lc[1].c_str());
      else {
        std::transform(lc[2].begin(), lc[2].end(), lc[2].begin(), ::tolower);
        if (lc[2] == "s") system.constants.md_ft = atof(lc[1].c_str())*1e15;
        else if (lc[2] == "ms") system.constants.md_ft = atof(lc[1].c_str())*1e12;
        else if (lc[2] == "us") system.constants.md_ft = atof(lc[1].c_str())*1e9;
        else if (lc[2] == "ns") system.constants.md_ft = atof(lc[1].c_str())*1e6;
        else if (lc[2] == "ps") system.constants.md_ft = atof(lc[1].c_str())*1e3;
        else if (lc[2] == "fs") system.constants.md_ft = atof(lc[1].c_str());
        else system.constants.md_ft = atof(lc[1].c_str());
      }
      std::cout << "Got MD final step = " << system.constants.md_ft << " fs";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "md_rotations")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.md_rotations = 1;
      else system.constants.md_rotations = 0;
      std::cout << "Got MD rotations option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "md_translations")) {
      if (!strcasecmp(lc[1].c_str(),"off")) system.constants.md_translations = 0;
      std::cout << "Got MD translations option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "thermostat")) {
      std::transform(lc[1].begin(), lc[1].end(), lc[1].begin(), ::tolower);
      if (lc[1] == "andersen")
        system.constants.thermostat_type = THERMOSTAT_ANDERSEN;
      else if (lc[1] == "nose-hoover")
        system.constants.thermostat_type = THERMOSTAT_NOSEHOOVER;
      std::cout << "Got Thermostat type = " << lc[1].c_str();
      printf("\n");
    } else if (!strcasecmp(lc[0].c_str(), "nh_q")) {
      system.constants.NH_Q = atof(lc[1].c_str());
      system.constants.user_Q = 1; // flag so we don't calculate a default Q later
      std::cout << "Got Nose Hoover Q parameter = " << lc[1].c_str() << " K fs^2";
      printf("\n");
    } else if (!strcasecmp(lc[0].c_str(), "nh_q_scale")) {
      system.constants.NH_Q_scale = atof(lc[1].c_str());
      std::cout << "Got Nose Hoover Q scale parameter = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "sig_override")) {
      system.constants.sig_override[lc[1]] = atof(lc[2].c_str());
      std::cout << "Got LJ sigma override for " << lc[1].c_str() << " = " << lc[2].c_str() << " A.";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "eps_override")) {
      system.constants.eps_override[lc[1]] = atof(lc[2].c_str());
      std::cout << "Got LJ epsilon override for " << lc[1].c_str() << " = " << lc[2].c_str() << " K.";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "charge_override")) {
      system.constants.charge_override[lc[1]] = atof(lc[2].c_str());
      std::cout << "Got charge override for " << lc[1].c_str() << " = " << lc[2].c_str() << " e.";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "lj_uff")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.lj_uff = 1;
      std::cout << "Got LJ UFF option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "vand_polar")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.polars_vand = 1;
      std::cout << "Got van Duijnen polarizability option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "methane_nist_fugacity")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.methane_nist_fugacity = 1;
      std::cout << "Got CH4 NIST Fugacity calculation option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "radial_dist") || !strcasecmp(lc[0].c_str(), "radial_distribution")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.stats.radial_dist = 1;
      else system.stats.radial_dist = 0;
      std::cout << "Got radial distribution option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "radial_bin_size")) {
      system.stats.radial_bin_size = atof(lc[1].c_str());
      std::cout << "Got radial bin size = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "radial_max_dist")) {
      system.stats.radial_max_dist = atof(lc[1].c_str());
      std::cout << "Got radial maximum distance = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "radial_centroid")) {
      for (int i=0; i<((int)lc.size()-1); i++) {
        if (lc[i+1] == "!") break;
        system.stats.radial_centroid.push_back( lc[i+1].c_str() );
        std::cout << "Got radial centroid[" << i << "] = " << lc[i+1].c_str();
        printf("\n");
      }

    } else if (!strcasecmp(lc[0].c_str(), "radial_counterpart")) {
      for (int i=0; i<((int)lc.size()-1); i++) {
        if (lc[i+1] == "!") break;
        system.stats.radial_counterpart.push_back( lc[i+1].c_str() );
        std::cout << "Got radial counterpart[" << i << "] = " << lc[i+1].c_str();
        printf("\n");
      }

    } else if (!strcasecmp(lc[0].c_str(), "radial_file")) {
      system.stats.radial_file = lc[1].c_str();
      std::cout << "Got radial dist. file = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "radial_exclude_molecules")) {
      if (!strcasecmp(lc[1].c_str(),"off")) system.stats.radial_exclude_molecules = 0;
      std::cout << "Got radial distribution exclusion of intramolecular pairs = " << lc[1].c_str();
      printf("\n");
    } else if (!strcasecmp(lc[0].c_str(), "checkpoints_option") || !strcasecmp(lc[0].c_str(), "checkpoints")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.checkpoints_option = 1;
      else system.constants.checkpoints_option = 0;
      std::cout << "Got checkpoints option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "total_energy")) {
      system.constants.total_energy = atof(lc[1].c_str());
      std::cout << "Got NVE total energy constant = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "rd_lrc")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.rd_lrc = 1;
      else system.constants.rd_lrc = 0;
      std::cout << "Got RD long range correction option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "ewald_es")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.ewald_es = 1;
      else system.constants.ewald_es = 0;
      std::cout << "Got Ewald electrostatics option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "kspace_option")) {
      if (!strcasecmp(lc[1].c_str(),"off")) system.constants.kspace_option = 0;
      std::cout << "Got Ewald Force k-space option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "ewald_kmax")) {
      system.constants.ewald_kmax = atof(lc[1].c_str());
      std::cout << "Got Ewald kmax = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "pdb_long")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.pdb_long =1;
      else system.constants.pdb_long = 0;
      std::cout << "Got option for PDB long float output = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "manual_cutoff")) {
      system.constants.manual_cutoff = 1;
      system.constants.manual_cutoff_val = atof(lc[1].c_str());
      std::cout << "Got manual pair-interaction spherical cutoff = " << lc[1].c_str() << " A.";
      printf("\n");
    } else if (!strcasecmp(lc[0].c_str(), "crystalbuild")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.crystalbuild = 1;
      std::cout << "Got crystal-builder option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "crystalbuild_includemovers")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.crystalbuild_includemovers = 1;
      std::cout << "Got crystal-builder include movable molecules option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "crystalbuild_x")) {
      system.constants.crystalbuild_x = atoi(lc[1].c_str());
      std::cout << "Got crystal-builder in x = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "crystalbuild_y")) {
      system.constants.crystalbuild_y = atoi(lc[1].c_str());
      std::cout << "Got crystal-builder in y = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "crystalbuild_z")) {
      system.constants.crystalbuild_z = atoi(lc[1].c_str());
      std::cout << "Got crystal-builder in z = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "scale_charges")) {
      system.constants.scale_charges = 1;
      system.constants.scale_charges_factor = atof(lc[1].c_str());
      std::cout << "Got scale-charges option = on; factor = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "makefrags")) {
      system.constants.fragmaker = 1;
      system.constants.numfrags = atoi(lc[1].c_str());
      std::cout << "Got fragment-maker = on. Going to make " << lc[1] << " base fragments.";
      printf("\n");
    } else if (!strcasecmp(lc[0].c_str(), "atoms_per_frag")) {
      system.constants.fragsize.clear(); // empty the default (250)
      for (int x=0; x<(int)lc.size()-1; x++) {
        if (lc[x+1] == "!") break; // avoid tailing comments
        system.constants.fragsize.push_back(atoi(lc[x+1].c_str()));
        std::cout << "Got atoms-per-fragment[" << x+1 << "] = " << lc[x+1].c_str();
        printf("\n");
      }

    } else if (!strcasecmp(lc[0].c_str(), "frag_bondlength")) {
      system.constants.frag_bondlength = atof(lc[1].c_str());
      std::cout << "Got fragment (initial/default) bond-length = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "calc_pressure_option")) {
      if (!strcasecmp(lc[1].c_str(),"off")) system.constants.calc_pressure_option = 0;
      std::cout << "Got calculate pressure option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "charge_sum_check")) {
      if (!strcasecmp(lc[1].c_str(),"off")) system.constants.charge_sum_check = 0;
      std::cout << "Got charge sum check option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "bondlength")) {
      system.constants.bondlength = atof(lc[1].c_str());
      std::cout << "Got max-bondlength parameter (for dynamic bond detection) = " << lc[1].c_str() << " A.";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "input_structure_ff")) {
      if (!strcasecmp(lc[1].c_str(),"on")) system.constants.input_structure_FF = 1;
      std::cout << "Got crystal-based forcefield option = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "opt_convergence") || !strcasecmp(lc[0].c_str(), "opt_error")) {
      system.constants.opt_error = atof(lc[1].c_str());
      std::cout << "Got optimization convergence = " << lc[1].c_str() << " kcal/mol.";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "opt_step_limit")) {
      system.constants.opt_step_limit = atoi(lc[1].c_str());
      std::cout << "Got optimization step limit = " << lc[1].c_str() << " steps.";
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "opt_mode")) {
      std::transform(lc[1].begin(), lc[1].end(), lc[1].begin(), ::tolower);
      if (lc[1] == "mc")
        system.constants.opt_mode = OPTIMIZE_MC;
      else if (lc[1] == "sd")
        system.constants.opt_mode = OPTIMIZE_SD;
      std::cout << "Got optimization mode = " << lc[1].c_str();
      printf("\n");
    } else if (!strcasecmp(lc[0].c_str(), "opt_bonds")) {
      if (!strcasecmp(lc[1].c_str(),"off"))
        system.constants.opt_bonds = 0;
      std::cout << "Got optimization bond contributions = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "opt_angles")) {
      if (!strcasecmp(lc[1].c_str(),"off"))
        system.constants.opt_angles = 0;
      std::cout << "Got optimization angle contributions = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "opt_dihedrals")) {
      if (!strcasecmp(lc[1].c_str(),"off"))
        system.constants.opt_dihedrals = 0;
      std::cout << "Got optimization dihedral angle contributions = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "opt_LJ")) {
      if (!strcasecmp(lc[1].c_str(),"off"))
        system.constants.opt_LJ = 0;
      std::cout << "Got optimization LJ non-bonding contributions = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "opt_ES")) {
      if (!strcasecmp(lc[1].c_str(),"off"))
        system.constants.opt_ES = 0;
      std::cout << "Got optimization ES non-bonding contributions = " << lc[1].c_str();
      printf("\n");

    } else if (!strcasecmp(lc[0].c_str(), "flexible_frozen")) {
      if (!strcasecmp(lc[1].c_str(),"on")) {
        system.constants.flexible_frozen = 1;
        system.constants.xyz_traj_movers_option=0; // we have to do this so the trajectory (including MOF) gets written fully
        system.constants.pdb_bigtraj_option = 1;   // and this
      }
      std::cout << "Got flexible frozen molecule option = " << lc[1].c_str();
      printf("\n");

    } else {
      std::cout << "ERROR: INPUT COMMAND'" << lc[0].c_str() << "' UNRECOGNIZED.";
      printf("\n");
      exit(EXIT_FAILURE);
    }
  } // end while reading lines
  printf("Done reading input parameters.\n\n");
} // end read input function

void paramOverrideCheck(System &system) {
  // LJ sigma/eps override if needed.
  if ((int)system.constants.sig_override.size() > 0) {
    map<string, double>::iterator it;
    for ( it = system.constants.sig_override.begin(); it != system.constants.sig_override.end(); it++ ) {
      for (int i=0; i<system.molecules.size(); i++) {
        for (int j=0; j<system.molecules[i].atoms.size(); j++) {
          if (system.molecules[i].atoms[j].name == it->first)
            system.molecules[i].atoms[j].sig = it->second;
        }
      }
    } // end map loop
  } // end if sigma overrides

  if ((int)system.constants.eps_override.size() > 0) {
    map<string, double>::iterator it;
    for (it = system.constants.eps_override.begin(); it != system.constants.eps_override.end(); it++) {
      for (int i=0; i<system.molecules.size(); i++) {
        for (int j=0; j<system.molecules[i].atoms.size(); j++) {
          if (system.molecules[i].atoms[j].name == it->first)
            system.molecules[i].atoms[j].eps = it->second;
        }
      }
    } // end map loop
  } // end if epsilon override

  if ((int)system.constants.charge_override.size() > 0) {
    map<string, double>::iterator it;
    for (it = system.constants.charge_override.begin(); it != system.constants.charge_override.end(); it++) {
      for (int i=0; i<system.molecules.size(); i++) {
        for (int j=0; j<system.molecules[i].atoms.size(); j++) {
          if (system.molecules[i].atoms[j].name == it->first)
            system.molecules[i].atoms[j].C = it->second*system.constants.E2REDUCED;
        }
      }
    } // end map loop
  } // end if charge override

  // universal UFF LJ parameters override
  if (system.constants.lj_uff == 1) {
    for (int i=0; i<system.molecules.size(); i++) {
      for (int j=0; j<system.molecules[i].atoms.size(); j++) {
        system.molecules[i].atoms[j].sig = system.constants.sigs[system.molecules[i].atoms[j].name.c_str()];
        system.molecules[i].atoms[j].eps = system.constants.eps[system.molecules[i].atoms[j].name.c_str()];
      }
    }

  }

  // universal van Duijnen polarizability parameters
  if (system.constants.polars_vand == 1) {
    for (int i=0; i<system.molecules.size(); i++) {
      for (int j=0; j<system.molecules[i].atoms.size(); j++) {
        system.molecules[i].atoms[j].polar = system.constants.polars[system.molecules[i].atoms[j].name.c_str()];
      }
    }

  }
}

void write_dipole(System &system, int step) {

  int p,i,j;
  FILE * fp;
  double DEBYE2SKA = system.constants.DEBYE2SKA;

  double dipole[3];
  double mag;

  fp = fopen(system.constants.dipole_output.c_str(), "a");

  fprintf(fp, "Step %i : Molecular induced dipoles (in Debye) as sum of atomic dipoles from Thole-Applequist Polarization\n",step);
  fprintf(fp, "#id #name #ux #uy #uz #u_mag\n");
  for(i=0; i<system.molecules.size(); i++) {
    if (system.molecules[i].frozen) continue;
    for(p = 0; p < 3; p++) dipole[p] = 0;
    for(j=0; j<system.molecules[i].atoms.size(); j++) {
      for(p = 0; p < 3; p++)
        dipole[p] += system.molecules[i].atoms[j].dip[p];
    }
    mag = sqrt(dipole[0]*dipole[0] + dipole[1]*dipole[1] + dipole[2]*dipole[2])/DEBYE2SKA;
    fprintf(fp, "%i %s %f %f %f %f\n", i, system.molecules[i].name.c_str(), dipole[0]/DEBYE2SKA, dipole[1]/DEBYE2SKA, dipole[2]/DEBYE2SKA, mag);
  }
  fflush(fp);
  fclose(fp);
  return;
}

void write_molec_dipole(System &system, int step) {
  int p,i,j;
  FILE * fp;
  //double DEBYE2SKA = system.constants.DEBYE2SKA;
  double E2REDUCED = system.constants.E2REDUCED;
  double eA2D = system.constants.eA2D;
  double dipole[3];
  double mag;

  fp = fopen(system.constants.molec_dipole_output.c_str(), "a");
  fprintf(fp,"Step %i : Molecular instananeous dipoles (in Debye) by sum[(r-r_com)*q]\n",step);
  fprintf(fp, "#id #name #ux #uy #uz #u_mag\n");

  for (i=0; i<system.molecules.size(); i++) {
    if (system.molecules[i].frozen) continue;
    for (p=0; p<3; p++) dipole[p] = 0;
    system.molecules[i].calc_center_of_mass();
    for (j=0; j<system.molecules[i].atoms.size(); j++) {
      for (p=0; p<3; p++)
        dipole[p] += (system.molecules[i].atoms[j].C/E2REDUCED)*(system.molecules[i].atoms[j].pos[p] - system.molecules[i].com[p])*eA2D; // converted to D
    }
    mag = sqrt(dipole[0]*dipole[0] + dipole[1]*dipole[1] + dipole[2]*dipole[2]);
    fprintf(fp, "%i %s %f %f %f %f\n", i, system.molecules[i].name.c_str(), dipole[0], dipole[1], dipole[2], mag);
  }
  fflush(fp);
  fclose(fp);
  return;

}

void inputValidation(System &system) {
  // check all the input and atoms and stuff to make sure we don't
  // get errors because the user is a noob

  int e = system.constants.ensemble;
  int mcmd_flag;
  if (system.constants.mode == "sp" || system.constants.mode == "opt") mcmd_flag=0;
  else mcmd_flag=1;

  if (system.constants.mode != "mc" && system.constants.mode != "md" && system.constants.mode != "sp" && system.constants.mode != "opt") {
    std::cout << "ERROR: No valid mode specified. Use mode [mc, md, sp, opt]. e.g. `mode mc` for Monte Carlo simulation." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!mcmd_flag && system.molecules.size() != 1) {
    if (system.proto.size() != 1) {
      std::cout << "ERROR: You asked for a single-point energy (or optimization) but the input does not have 1 molecule. Make sure only 1 molecule is in your input atoms file, or use `sorbate_name h2o_tip4p`, for example. If you want energy for multiple molecules just use `mode mc` with `steps 0` and `mc_corrtime 0`.";
      printf("\n");
      exit(EXIT_FAILURE);
    }
  }
  if (mcmd_flag && (e != ENSEMBLE_NPT && e != ENSEMBLE_NVT && e != ENSEMBLE_UVT && e != ENSEMBLE_NVE)) {
    std::cout << "ERROR: No ensemble specified. Use   ensemble uvt   , for example." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (system.constants.mode == "mc" && !(system.constants.finalstep >= 0)) {
    std::cout << "ERROR: Monte carlo was activated but you need to specify finalstep, e.g.  finalstep 100000" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (mcmd_flag && ((e == ENSEMBLE_UVT || e == ENSEMBLE_NPT || e == ENSEMBLE_NVT ) && system.constants.temp == 0)) {
    std::cout << "ERROR: You specified an ensemble with constant T but didn't supply T (or you set T=0)." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (system.constants.mode == "md" && (e == ENSEMBLE_NPT)) {
    std::cout << "ERROR: You specified MD mode but selected an incompatible ensemble. MD supports uVT, NVT and NVE only." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (mcmd_flag && (system.pbc.a == 0 && system.pbc.b == 0 && system.pbc.c == 0 && system.pbc.alpha == 0 && system.pbc.beta ==0 && system.pbc.gamma == 0)) {
    std::cout << "ERROR: You didn't supply a basis to build the box. Even simulations with no PBC need a box." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (system.constants.mode != "md" && system.constants.gpu) {
    std::cout << "ERROR: GPU was enabled but simulation mode is not MD. GPU can only be enabled with MD.";
    exit(EXIT_FAILURE);
  }
  if (system.constants.bias_uptake != 0 && (system.constants.ensemble != ENSEMBLE_UVT || system.constants.mode != "mc" || system.proto.size() > 1)) {
    std::cout << "ERROR: Uptake bias option was used but single-sorbate uVT MC is not set.";
    exit(EXIT_FAILURE);
  }
  if (system.constants.mode == "md" && !system.constants.md_pbc) {
    std::cout << "ERROR: MD mode is on but MD PBC was set off. This feature is not available. (You must have a periodic box for MD). For example, use a huge box with 'manual_cutoff 10`, to get the effect of free space.";
    exit(EXIT_FAILURE);
  }
  if (system.constants.mode == "md" && system.constants.ensemble == ENSEMBLE_NVT && system.constants.thermostat_type == THERMOSTAT_NOSEHOOVER && system.constants.md_manual_init_vel == 1) {
    std::cout << "ERROR: You cannot use the Nose-Hoover (default) thermostat in NVT MD with a user-defined initial velocity (initial velocities must be determined automatically for this thermostat). Try deleting the line `md_init_vel XXXXX` in the input.";
    exit(EXIT_FAILURE);
  }
#ifndef GPU  // if no gpu compilation, error out if user is asking for GPU features
  if (system.constants.gpu) {
    std::cout << "ERROR: You set GPU feature on but did not compile with GPU macro enabled";
    exit(EXIT_FAILURE);
  }
#endif
  if (system.stats.radial_dist && (system.stats.radial_centroid.size() != system.stats.radial_counterpart.size())) {
    std::cout << "ERROR: The number of radial_centroid parameters is not equal to the number of radial_counterpart parameters.";
    exit(EXIT_FAILURE);
  }
  // charge sum check
  if (system.constants.charge_sum_check && mcmd_flag) {
    double qsum=0;
    for (int i=0; i<system.molecules.size(); i++)
      for (int j=0; j<system.molecules[i].atoms.size(); j++)
        qsum += system.molecules[i].atoms[j].C;

    if (fabs(qsum/system.constants.E2REDUCED) > 0.005 && // a little bit of lee-way for charge sum.
        (system.constants.mode == "md" || system.constants.mc_pbc) &&
        (system.constants.potential_form != POTENTIAL_LJ && system.constants.potential_form != POTENTIAL_COMMY)
       )
    {
      std::cout << "ERROR: The sum of charges (" << qsum/system.constants.E2REDUCED 
                << " e) on atoms is not zero. The system must be neutral for ewald summation to be correct.";
      exit(EXIT_FAILURE);
    }
  } // end q-check

  if (system.constants.mode == "md" && system.constants.ensemble != ENSEMBLE_UVT && 
      system.stats.count_movables <= 0 && !(system.constants.flexible_frozen && system.stats.count_frozens > 0)) {
    std::cout << "ERROR: MD is turned on but there are no movable molecules in the input "
              << "(and ensemble is not uVT). (Use 'M' as opposed to 'F' to distinguish movers from frozens)";
    exit(EXIT_FAILURE);
  }
  // single-atom movers-only check for MD rotation
  int flag=0;
  for (int i=0; i<system.molecules.size(); i++) {
    for (int j=0; j<system.molecules[i].atoms.size(); j++) {
      if (!system.molecules[i].atoms[j].frozen && system.molecules[i].atoms.size() > 1) flag=1;
    }
  }
  if (system.constants.mode == "md" && system.constants.md_rotations && !flag && e != ENSEMBLE_UVT) {
    std::cout << "ERROR: MD rotations were turned on but there are no movable molecules with >1 atom in input. "
              << "Turn md_rotations off.";
    exit(EXIT_FAILURE);
  }
  if (system.constants.crystalbuild && (
        (system.constants.crystalbuild_x == 1 && system.constants.crystalbuild_y ==1 && system.constants.crystalbuild_z == 1) || (
          fabs(system.constants.crystalbuild_x) < 1 || fabs(system.constants.crystalbuild_y) < 1 || fabs(system.constants.crystalbuild_z) < 1)
        )
     ) {
    std::cout << "ERROR: You turned the crystal-builder on but did not specify a (correct) dimension to duplicate. e.g., `crystalbuild_y 2` is acceptable but `crystalbuild_y 0` is not. Leave the option off if you don't want to use it. (i.e. x y z are set to 1 by default).";
    exit(EXIT_FAILURE);
  }
  if (mcmd_flag && e == ENSEMBLE_UVT && system.stats.count_movables <= 0 && system.constants.sorbate_name.size() ==0) {
    std::cout << "ERROR: You specified uVT with a system containing no movable molecules and did not specify a desired sorbate. Try `sorbate_name h2_bssp`, for example.";
    exit(EXIT_FAILURE);
  }
  // check 0 value for all c6,c8 params for Tang-Toennies.
  flag = 0;
  for (int i=0; i<system.molecules.size(); i++) {
    for (int j=0; j<system.molecules[i].atoms.size(); j++) {
      if (system.molecules[i].atoms[j].c6 != 0 || system.molecules[i].atoms[j].c8 != 0) flag=1;
    }
  }
  if ( (system.constants.potential_form == POTENTIAL_TT || system.constants.potential_form == POTENTIAL_TTES || system.constants.potential_form == POTENTIAL_TTESPOLAR) && !flag) {
    std::cout << "ERROR: Tang-Toennies potential was requested but the C6/C8 terms for all atoms are zero. In an input .pdb, column 16 = C6, column 17 = C8, and column 18 = C10 (optional). All expressed in atomic units.";
    exit(EXIT_FAILURE);
  }
  // check for a box
  int nobox = 1;
  for (int p=0; p<3; p++)
    for (int q=0; q<3; q++)
      if (system.pbc.basis[p][q] != 0.0) nobox = 0;
  if (nobox && mcmd_flag) {
    std::cout << "ERROR: No box information was supplied. Use `carbasis [a] [b] [c] [alpha] [beta] [gamma]`, for example.";
    exit(EXIT_FAILURE);
  }
  if (system.constants.histogram_option==1 && (int)system.proto.size() > 1) {
    std::cout << "ERROR: Histogram is currently only available for single-sorbate systems. Use `histogram off` to fix this error.";
    exit(EXIT_FAILURE);
  }
  if (system.constants.flexible_frozen && system.constants.mode != "md") {
    std::cout << "ERROR: flexible frozen option is only available for MD simulations.";
    exit(EXIT_FAILURE);
  }
  if (system.constants.mode=="md" && system.stats.count_movables == 0 && system.constants.md_mode == MD_MOLECULAR && system.constants.ensemble != ENSEMBLE_UVT) {
    std::cout << "ERROR: `md_mode` is set to `molecular` but there are no movable molecules in the input atoms file. Did you mean to use `md_mode flexible` (e.g. to simulate a flexible MOF by itself), or `ensemble uvt` (e.g. to allow addition of movable molecules)?";
    exit(EXIT_FAILURE);
  }
#ifndef OMP
  if (system.constants.openmp_threads > 0) {
    std::cout << "ERROR: You used the `omp` command but did not compile with OpenMP resources. Compile with, e.g. `bash compile.sh omp linux`.";
    exit(EXIT_FAILURE);
  }
#endif
  if (system.constants.mode=="md" && system.constants.ensemble != ENSEMBLE_NVT && system.constants.calc_pressure_option) {
    system.constants.calc_pressure_option=0;
    std::cout << "Turned off calculate-pressure option (only valid for NVT MD).";
    printf("\n");
  }
  if (system.constants.integrator == INTEGRATOR_RK4) {
    std::cout << "ERROR: RK4 integration is disabled (under development). Use the default velocity verlet by `integrator vv`.";
    printf("\n");
    exit(EXIT_FAILURE);
  }
  if (system.constants.md_rotations && system.constants.md_mode==MD_FLEXIBLE) {
    std::cout << "WARNING: Can't use `md_rotations on` with `md_mode flexible`. Disabling MD rotations now.";
    printf("\n");
    system.constants.md_rotations=0;
  }
}
// end input validation function
