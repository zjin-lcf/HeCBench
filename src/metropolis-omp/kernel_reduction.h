//  trueke                                                                      //
//  A multi-GPU implementation of the exchange Monte Carlo method.              //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2015 Cristobal A. Navarro, Wei Huang.                           //
//                                                                              //
//  This file is part of trueke.                                                //
//  trueke is free software: you can redistribute it and/or modify              //
//  it under the terms of the GNU General Public License as published by        //
//  the Free Software Foundation, either version 3 of the License, or           //
//  (at your option) any later version.                                         //
//                                                                              //
//  trueke is distributed in the hope that it will be useful,                   //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of              //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               //
//  GNU General Public License for more details.                                //
//                                                                              //
//  You should have received a copy of the GNU General Public License           //
//  along with trueke.  If not, see <http://www.gnu.org/licenses/>.             //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
#ifndef _REDUCTION_H_
#define _REDUCTION_H_


/* energy reduction using block reduction */
template <typename T>
void kernel_redenergy(const int *s, int L, T *out, const int *H, float h)
{
  float energy = 0.f; 
  #pragma omp target teams distribute parallel for collapse(3) reduction(+:energy)
  for (int z = 0; z < L; z++)
    for (int y = 0; y < L; y++)
      for (int x = 0; x < L; x++) {
        int id = C(x,y,z,L);
        // this optimization only works for L being a power of 2
        //float sum = -(float)(s[id] * ((float)(s[C((x+1) & (L-1), y, z, L)] + 
        // s[C(x, (y+1) & (L-1), z, L)] + s[C(x, y, (z+1) & (L-1), L)]) + h*H[id]));

        // this line works always
        float sum = -(float)(s[id] * ((float)(s[C((x+1) >=  L? 0: x+1, y, z, L)] + 
                    s[C(x, (y+1) >= L? 0 : y+1, z, L)] + s[C(x, y, (z+1) >= L? 0 : z+1, L)]) + h*H[id]));

        energy += sum;
      }

  *out = energy;
}


#endif
