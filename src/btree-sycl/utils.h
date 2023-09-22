/*Copyright(c) 2020, The Regents of the University of California, Davis.            */
/*                                                                                  */
/*                                                                                  */
/*Redistribution and use in source and binary forms, with or without modification,  */
/*are permitted provided that the following conditions are met :                    */
/*                                                                                  */
/*1. Redistributions of source code must retain the above copyright notice, this    */
/*list of conditions and the following disclaimer.                                  */
/*2. Redistributions in binary form must reproduce the above copyright notice,      */
/*this list of conditions and the following disclaimer in the documentation         */
/*and / or other materials provided with the distribution.                          */
/*                                                                                  */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   */
/*ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED     */
/*WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.*/
/*IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,  */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT */
/*NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR*/
/*PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, */
/*WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) */
/*ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE        */
/*POSSIBILITY OF SUCH DAMAGE.                                                       */
/************************************************************************************/

#pragma once

#include <iostream>

namespace memoryUtil {
template <typename DataT>
void cpyToHost(sycl::queue &q, DataT *&src_data, DataT *dst_data, std::size_t count) {
  q.memcpy(dst_data, src_data, sizeof(DataT) * count).wait();
}

template <typename DataT>
void cpyToDevice(sycl::queue &q, DataT *src_data, DataT *&dst_data, std::size_t count) {
  q.memcpy(dst_data, src_data, sizeof(DataT) * count).wait();
}

template <typename DataT>
void deviceAlloc(sycl::queue &q, DataT *&src_data, std::size_t count) {
  src_data = (DataT *)sycl::malloc_device(sizeof(DataT) * count, q);
}

template <typename DataT, typename ByteT>
void deviceSet(sycl::queue &q, DataT *&src_data, std::size_t count, ByteT value) {
  q.memset(src_data, value, sizeof(DataT) * count).wait();
}

template <typename DataT>
void deviceFree(sycl::queue &q, DataT *src_data) {
  sycl::free(src_data, q);
}
}  // namespace memoryUtil

#define LANEID_REVERSED(laneId) (31 - laneId)

inline unsigned lane_id(sycl::nd_item<1> &item) {
  return item.get_local_id(0) & 0x1F;
}
