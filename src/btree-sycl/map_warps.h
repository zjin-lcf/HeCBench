/*Copyriitemght(c) 2020, The Regents of the University of California, Davis.
 */
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

namespace GpuBTree {
namespace warps {
#define KEY_PIVOT_MASK 0xAAAAAAA8
#define PIVOT_KEY_MASK 0x55555554
#define getAddressPtr(address) (d_pool + address * 32)
#define allocate() \
   sycl::atomic<uint32_t>(sycl::global_ptr<uint32_t>(d_count)).fetch_add(1)

inline uint32_t volatileRead(uint32_t *address) {
  uint32_t data;
  data = *address;

  return data;
}

inline void volatileWrite(uint32_t *address, uint32_t data) {
  *address = data;
}

inline uint32_t volatileNodeReadR(uint32_t *nodeAddress, sycl::nd_item<1> &item) {
  return volatileRead(nodeAddress + LANEID_REVERSED(lane_id(item)));
}

inline uint32_t volatileNodeRead(uint32_t *nodeAddress, sycl::nd_item<1> &item) {
  return volatileRead(nodeAddress + lane_id(item));
}

inline void volatileNodeWriteR(uint32_t *nodeAddress, uint32_t data,
                                        sycl::nd_item<1> &item) {
  volatileWrite(nodeAddress + LANEID_REVERSED(lane_id(item)), data);
}

inline void volatileNodeWrite(uint32_t *nodeAddress, uint32_t data,
                                       sycl::nd_item<1> &item) {
  volatileWrite(nodeAddress + lane_id(item), data);
}

bool try_acquire_lock(uint32_t* nodeAddress, sycl::nd_item<1> &item) {
  bool isLocked = true;
  {
    if (lane_id(item) == 1) {
      isLocked =
          (sycl::atomic<uint32_t>(sycl::global_ptr<uint32_t>(nodeAddress + 1))
               .fetch_or(0x80000000) & 0x80000000) != 0;
    }
    isLocked = sycl::select_from_group(item.get_sub_group(), isLocked, 1);
  }
  if (isLocked)
    return true;
  sycl::atomic_fence(sycl::memory_order::acq_rel,
                     sycl::memory_scope::device);
  return false;
}

inline int __ffs(int x) {
  return (x == 0) ? 0 : sycl::ext::intel::ctz(x) + 1;
}

void acquire_lock(uint32_t* nodeAddress, sycl::nd_item<1> &item) {
  bool isLocked = true;
  while (isLocked) {
    if (lane_id(item) == 1) {
      isLocked =
          (sycl::atomic<uint32_t>(sycl::global_ptr<uint32_t>(nodeAddress + 1))
               .fetch_or(0x80000000) & 0x80000000) != 0;
    }
    isLocked = sycl::select_from_group(item.get_sub_group(), isLocked, 1);
  }
  sycl::atomic_fence(sycl::memory_order::acq_rel,
                     sycl::memory_scope::device);
}

void release_lock(uint32_t* nodeAddress, sycl::nd_item<1> &item) {
  sycl::atomic_fence(sycl::memory_order::acq_rel,
                     sycl::memory_scope::device);
  if (lane_id(item) == 1) {
    sycl::atomic<uint32_t>(sycl::global_ptr<uint32_t>(nodeAddress + 1))
        .fetch_and(0x7fffffff);
  }
}

inline void
insert_into_node(bool isIntermediate, uint32_t next, uint32_t src_key,
                 uint32_t &src_value, uint32_t *d_pool,
                 sycl::nd_item<1> &item, uint32_t src_unit_data = 0) {
  // do we split?
  if (isIntermediate)
    src_unit_data = volatileNodeReadR(getAddressPtr(next), item);

  src_unit_data &= 0x7FFFFFFF;
  src_key = src_key & 0x7FFFFFFF;

  auto sg = item.get_sub_group();
  uint32_t insertion_location =
      sycl::reduce_over_group(
          sg, (src_key > src_unit_data) && (src_unit_data != 0)
              ? (0x1 << sg.get_local_linear_id()) : 0,
          sycl::ext::oneapi::plus<>()) &
      KEY_PIVOT_MASK;
  int dest_lane_pivot = __ffs(insertion_location) - 1;

  if (dest_lane_pivot < 0)
    dest_lane_pivot = 33;

  uint32_t to_move = (1U << (dest_lane_pivot - 3)) - 1U;
  uint32_t lane_mask = (1 << lane_id(item));

  uint32_t my_new_data =
      sycl::shift_group_left(sg, src_unit_data, 2); // move everything up two spaces

  to_move = to_move & 0xfffffffc;  // ignore links

  bool shifted = lane_mask & to_move;
  my_new_data = shifted * my_new_data + (1 - shifted) * src_unit_data;

  uint32_t key_loc = (1 << (dest_lane_pivot - 2));
  uint32_t val_loc = (1 << (dest_lane_pivot - 3));

  if (key_loc & lane_mask)
    my_new_data = src_key;
  if (val_loc & lane_mask)
    my_new_data = src_value;

  int valid_location = sycl::reduce_over_group(
      sg, src_unit_data != 0 ? (0x1 << sg.get_local_linear_id()) : 0,
      sycl::ext::oneapi::plus<>());
  valid_location = valid_location >> 2;
  if (isIntermediate && (valid_location & lane_mask & KEY_PIVOT_MASK)) {
    my_new_data = my_new_data | 0x80000000;
  }
  if (lane_id(item) == 30) {
    my_new_data = (my_new_data | 0x80000000);
  }

  volatileWrite(getAddressPtr(next) + LANEID_REVERSED(lane_id(item)), my_new_data);
  src_unit_data = my_new_data & 0x7FFFFFFF;
}

inline bool split_node1(uint32_t myParent, uint32_t src_key,
                                 uint32_t &nodeIdx, uint32_t &mydata,
                                 uint32_t *d_pool, uint32_t *d_count,
                                 sycl::nd_item<1> &item) {
  auto sg = item.get_sub_group(); 
  uint32_t rightData = sycl::shift_group_right(sg, mydata, 16);

  if (lane_id(item) < 18)
    rightData = 0;  // destroy all data except link
  if (lane_id(item) < 2)
    rightData = mydata;

  uint32_t rightDataMin = sycl::select_from_group(sg, rightData, 31) & 0x7FFFFFFF;

  acquire_lock(getAddressPtr(myParent), item);

  uint32_t parent_data = volatileNodeReadR(getAddressPtr(myParent), item);

  uint32_t parent_last_key = sycl::select_from_group(sg, parent_data, 3); // get last key

  if (parent_last_key)  // parent is full
  {
    release_lock(getAddressPtr(nodeIdx), item);
    release_lock(getAddressPtr(myParent), item);

    return true;
  }

  uint32_t isItMyParent =
      sycl::reduce_over_group(
          sg, (parent_data & 0x7FFFFFFF) == nodeIdx
              ? (0x1 << sg.get_local_linear_id()) : 0,
          sycl::ext::oneapi::plus<>()) &
      PIVOT_KEY_MASK;
  if (isItMyParent == 0)  // not my parent
  {
    release_lock(getAddressPtr(nodeIdx), item);
    release_lock(getAddressPtr(myParent), item);

    return true;
  }

  uint32_t rightIdx;
  if (!lane_id(item)) {
    rightIdx = allocate();
  }

  rightIdx = sycl::select_from_group(sg, rightIdx, 0);
  acquire_lock(getAddressPtr(rightIdx), item);

  // update parent
  insert_into_node(true, myParent, rightDataMin, rightIdx, d_pool, item,
                   parent_data);

  release_lock(getAddressPtr(myParent), item);

  if (lane_id(item) == 30)
    rightData |= 0x80000000;
  volatileWrite(getAddressPtr(rightIdx) + LANEID_REVERSED(lane_id(item)),
                rightData);

  // destroy upper 16 keys
  if (lane_id(item) < 16)
    mydata = 0;

  // add link
  if (lane_id(item) == 1)
    mydata = rightDataMin;
  if (lane_id(item) == 0)
    mydata = rightIdx;

  if (lane_id(item) == 30)
    mydata |= 0x80000000;

  volatileWrite(getAddressPtr(nodeIdx) + LANEID_REVERSED(lane_id(item)), mydata);

  if (src_key >= rightDataMin)  // go right
  {
    release_lock(getAddressPtr(nodeIdx), item);

    mydata = rightData;
    nodeIdx = rightIdx;
  } else {
    release_lock(getAddressPtr(rightIdx), item);
  }
  mydata &= 0x7FFFFFFF;

  return false;
}

inline void split_root_node(uint32_t src_key, uint32_t &nodeIdx,
                            uint32_t &mydata, 
                            uint32_t *d_pool, uint32_t *d_count,
                            sycl::nd_item<1> &item) {
  uint32_t leftIdx, rightIdx;

  if (!lane_id(item)) {
    leftIdx = allocate();
    rightIdx = allocate();
  }

  auto sg = item.get_sub_group();
  leftIdx = sycl::select_from_group(sg, leftIdx, 0);
  rightIdx = sycl::select_from_group(sg, rightIdx, 0);

  acquire_lock(getAddressPtr(leftIdx), item);
  acquire_lock(getAddressPtr(rightIdx), item);

  // update root
  uint32_t rootData;
  rootData = sycl::select_from_group(sg, mydata, 15) | 0x80000000;
  if (LANEID_REVERSED(lane_id(item)) == 0)
    rootData = mydata | 0x80000000;
  else if (LANEID_REVERSED(lane_id(item)) == 1)
    rootData = leftIdx | 0x80000000;  /// locked
  else if (LANEID_REVERSED(lane_id(item)) == 3)
    rootData = rightIdx;
  else if (LANEID_REVERSED(lane_id(item)) == 2)
    rootData = rootData;
  else
    rootData = 0;

  volatileWrite(getAddressPtr(nodeIdx) + LANEID_REVERSED(lane_id(item)), rootData);
  release_lock(getAddressPtr(nodeIdx), item);

  uint32_t rightDataMin = sycl::select_from_group(sg, rootData, 29) & 0x7fffffff;

  uint32_t leftData = mydata;
  if (lane_id(item) < 16) // left node
    leftData = 0;
  if (lane_id(item) == 1)
    leftData = rightDataMin;
  if (lane_id(item) == 0)
    leftData = rightIdx;

  if (lane_id(item) < 16) {
    if (lane_id(item) == 14)
      mydata |= 0x80000000;
    volatileWrite(getAddressPtr(rightIdx) + +15 - lane_id(item),
                  mydata);
  }

  if (lane_id(item) == 30)
    leftData |= 0x80000000;

  volatileWrite(getAddressPtr(leftIdx) + LANEID_REVERSED(lane_id(item)), leftData);

  if (LANEID_REVERSED(lane_id(item)) == 0)
    rootData = (rootData & 0x7fffffff);
  if (LANEID_REVERSED(lane_id(item)) == 2)
    rootData = (rootData & 0x7fffffff);
  bool goRight = true;
  if (LANEID_REVERSED(lane_id(item)) == 2 && src_key < rootData)
    goRight = false;

  goRight = sycl::select_from_group(sg, goRight, 29);

  // now shuffle
  if (goRight) {
    release_lock(getAddressPtr(leftIdx), item);

    mydata = sycl::shift_group_right(sg, mydata, 16);
    if (lane_id(item) < 16)
      mydata = 0;
    nodeIdx = rightIdx;
  } else {
    release_lock(getAddressPtr(rightIdx), item);
    mydata = leftData;
    nodeIdx = leftIdx;
  }
  mydata &= 0x7FFFFFFF;
}

template <typename KeyT, typename ValueT>
void insertion_unit(bool &to_be_inserted, KeyT &myKey,
                    ValueT &myValue, uint32_t *d_root,
                    uint32_t *d_pool, uint32_t *d_count,
                    sycl::nd_item<1> &item) {
  uint32_t work_queue;
  uint32_t last_work_queue = 0;
  uint32_t rootAddress = *d_root;
  uint32_t parent = rootAddress;
  uint32_t next = rootAddress;
  auto sg = item.get_sub_group();

  while ((work_queue = sycl::reduce_over_group(
              sg, to_be_inserted ? (0x1 << sg.get_local_linear_id()) : 0,
              sycl::ext::oneapi::plus<>()))) {
    char FullLeafLinkRoot = 0;
    uint32_t src_key = sycl::select_from_group(sg, myKey, __ffs(work_queue) - 1);

    if (last_work_queue != work_queue) {
      next = rootAddress;
      parent = rootAddress;
    }

    uint32_t src_unit_data = volatileNodeReadR(getAddressPtr(next), item);

    uint32_t link_min = sycl::select_from_group(sg, src_unit_data, 1) & 0x7FFFFFFF; // get link min

    while (link_min && src_key >= link_min)  // traverse to right
    {
      next = sycl::select_from_group(sg, src_unit_data, 0) & 0x7FFFFFFF; // get link min
      src_unit_data = volatileNodeReadR(getAddressPtr(next), item);
      link_min = sycl::select_from_group(sg, src_unit_data, 1) & 0x7FFFFFFF; // get link min
      FullLeafLinkRoot |= 0x2;
    }

    uint32_t first_key = sycl::select_from_group(sg, src_unit_data, 31); // get first key
    FullLeafLinkRoot =
        ((first_key & 0x80000000) == 0) ? FullLeafLinkRoot | 0x4 : FullLeafLinkRoot & 0x3;

    // acquire lock for a leaf
    if (FullLeafLinkRoot & 0x4) {
      if (try_acquire_lock(getAddressPtr(next), item)) {
        next = parent;
        continue;
      }
      src_unit_data = volatileNodeReadR(getAddressPtr(next), item);

      first_key = sycl::select_from_group(sg, src_unit_data, 31); // still a leaf?
      FullLeafLinkRoot = ((first_key & 0x80000000) == 0) ? FullLeafLinkRoot | 0x4
                                                         : FullLeafLinkRoot & 0x3;

      link_min = sycl::select_from_group(sg, src_unit_data, 1) & 0x7FFFFFFF; // get link min

      if ((parent == rootAddress) && link_min && src_key >= link_min) {
        release_lock(getAddressPtr(next), item);

        next = rootAddress;
        parent = rootAddress;
        continue;
      }

      if (!(FullLeafLinkRoot & 0x4))  // no, release the lock, traverse if needed
        release_lock(getAddressPtr(next), item);

      while (link_min && src_key >= link_min) {
        if (FullLeafLinkRoot & 0x4)
          release_lock(getAddressPtr(next), item);
        next = sycl::select_from_group(sg, src_unit_data, 0) & 0x7FFFFFFF; // get link min
        if (FullLeafLinkRoot & 0x4)
          acquire_lock(getAddressPtr(next), item);

        src_unit_data = volatileNodeReadR(getAddressPtr(next), item);

        first_key = sycl::select_from_group(sg, src_unit_data, 31); // still a leaf?
        FullLeafLinkRoot = ((first_key & 0x80000000) == 0) ? FullLeafLinkRoot | 0x4
                                                           : FullLeafLinkRoot & 0x3;

        if (!(FullLeafLinkRoot & 0x4))  // no, release the lock, traverse if needed
          release_lock(getAddressPtr(next), item);

        link_min = sycl::select_from_group(sg, src_unit_data, 1) &
                   0x7FFFFFFF; // get link min
                               // link_used = true;
        FullLeafLinkRoot |= 0x2;
      }
    }

    // parent info is correct
    FullLeafLinkRoot = sycl::select_from_group(sg, src_unit_data, 3)
            ? FullLeafLinkRoot | 0x8
            : FullLeafLinkRoot & 0x7;
    if ((FullLeafLinkRoot & 0x2) && (FullLeafLinkRoot & 0x8)) {
      if (FullLeafLinkRoot & 0x4) {
        release_lock(getAddressPtr(next), item);
        next = rootAddress;
        parent = rootAddress;
        continue;
      } else {
        FullLeafLinkRoot &= 0x7;
      }
    }

    if ((FullLeafLinkRoot & 0x8) &&
        !(FullLeafLinkRoot & 0x4))  // if it's leaf then it's locked already
    {
      if (try_acquire_lock(getAddressPtr(next), item)) {
        next = parent;
        continue;
      } else {
        src_unit_data =
            volatileNodeReadR(getAddressPtr(next), item);

        FullLeafLinkRoot =
            sycl::select_from_group(sg, src_unit_data, 3)
                ? FullLeafLinkRoot | 0x8
                : FullLeafLinkRoot & 0x7;

        if (FullLeafLinkRoot & 0x8)  // not full anymore?
        {
          link_min =
              sycl::select_from_group(sg, src_unit_data, 1) & 0x7FFFFFFF; // get link min
          if (link_min && src_key >= link_min)  // traverse to right?
          {
            release_lock(getAddressPtr(next), item);
            next = rootAddress;
            parent = rootAddress;
            continue;
          }
        } else {
          release_lock(getAddressPtr(next), item);

          link_min = sycl::select_from_group(sg, src_unit_data, 1) & 0x7FFFFFFF; // get link min
          while (link_min && src_key >= link_min)  // traverse to right
          {
            next = sycl::select_from_group(sg, src_unit_data, 0) & 0x7FFFFFFF; // get link min
            src_unit_data = volatileNodeReadR(getAddressPtr(next), item);
            link_min = sycl::select_from_group(sg, src_unit_data, 1) & 0x7FFFFFFF; // get link min
            FullLeafLinkRoot |= 0x2;
          }
        }
      }
    }

    if ((FullLeafLinkRoot & 0x8) && (next != rootAddress) && (parent == next)) {
      release_lock(getAddressPtr(next), item);
      next = rootAddress;
      parent = rootAddress;
      continue;
    }

    if ((FullLeafLinkRoot & 0x8) && (next != rootAddress)) {
      if (split_node1(parent, src_key, next, src_unit_data, d_pool, d_count,
                      item)) {
        next = rootAddress;
        parent = rootAddress;
        continue;
      }
      if (!(FullLeafLinkRoot & 0x4))
        release_lock(getAddressPtr(next), item);
    } else if ((FullLeafLinkRoot & 0x8)) {
      split_root_node(src_key, next, src_unit_data, d_pool, d_count, item);
      FullLeafLinkRoot |= 0x1;
      if (!((FullLeafLinkRoot & 0x4)))
        release_lock(getAddressPtr(next), item);
    }
    parent = (FullLeafLinkRoot & 0x1) ? rootAddress : next;
    if (FullLeafLinkRoot & 0x4) {
      uint32_t src_lane1 = __ffs(work_queue) - 1;
      uint32_t src_value = sycl::select_from_group(sg, myValue, src_lane1);
      bool key_exist =
          sycl::reduce_over_group(
              sg, src_key == src_unit_data
                  ? (0x1 << sg.get_local_linear_id()) : 0,
              sycl::ext::oneapi::plus<>()) &
          KEY_PIVOT_MASK;
      if (!key_exist)
        insert_into_node(false, next, src_key, src_value, d_pool, item,
                         src_unit_data);
      release_lock(getAddressPtr(next), item);
      if (src_lane1 == lane_id(item))
        to_be_inserted = false;
    } else {
      src_unit_data = src_unit_data ? src_unit_data : 0xFFFFFFFF;
      uint32_t src_unit_key = src_unit_data & 0x7FFFFFFF;

      next = sycl::reduce_over_group(
                 sg, src_key >= (src_unit_key)
                     ? (0x1 << sg.get_local_linear_id()) : 0,
                 sycl::ext::oneapi::plus<>()) &
             KEY_PIVOT_MASK;
      next = __ffs(next);
      if (next == 0)
        next = sycl::select_from_group(sg, src_unit_key, 30);
      else
        next = sycl::select_from_group(sg, src_unit_key, next - 2);
    }
    last_work_queue = work_queue;
  }
}

//////////////////
////// Search  ///
//////////////////
#define SEARCH_NOT_FOUND 0

void search_unit(bool &to_be_searched, uint32_t &laneId,
                 uint32_t &myKey, uint32_t &myResult,
                 uint32_t *d_root, uint32_t *d_pool,
                 sycl::nd_item<1> &item) {
  uint32_t rootAddress = *d_root;

  uint32_t landId_reversed = 31 - laneId;
  uint32_t work_queue = 0;
  uint32_t last_work_queue = 0;
  uint32_t next = rootAddress;  // starts from the root
  auto sg = item.get_sub_group();

  while ((work_queue = sycl::reduce_over_group(
              sg, to_be_searched ? (0x1 << sg.get_local_linear_id()) : 0,
              sycl::ext::oneapi::plus<>()))) {
    uint32_t src_lane = __ffs(work_queue) - 1;
    uint32_t src_key = sycl::select_from_group(sg, myKey, src_lane);

    bool found = false;
    next = (last_work_queue != work_queue)
               ? rootAddress
               : next;  // if previous round successful, we start from the root again

    uint32_t src_unit_data = *(getAddressPtr(next) + landId_reversed);

    bool isLeaf = ((src_unit_data & 0x80000000) == 0);  // only even lanes are valid
    isLeaf = sycl::select_from_group(sg, isLeaf, 31); 
                         // some pairs are invalid -- either pass all pairs in
                         // same level as leaves or just use the first element
                         // leaf status -- 31 because reversed order

    src_unit_data = src_unit_data ? src_unit_data
                                  : 0xFFFFFFFF;  // valid entry : make sure we end up with
                                                 // zero ballot bit for invalid ones

    uint32_t src_unit_key = src_unit_data & 0x7FFFFFFF;

    // looking for the right pivot, only valid at intermediate nodes
    uint32_t isFoundPivot_bmp =
        sycl::reduce_over_group(
            sg, src_key >= src_unit_key
                ? (0x1 << sg.get_local_linear_id()) : 0,
            sycl::ext::oneapi::plus<>()) &
        KEY_PIVOT_MASK;
    int dest_lane_pivot = __ffs(isFoundPivot_bmp) - 1;

    if (dest_lane_pivot < 0) {  // not found in an intermediate node
      if (laneId == src_lane) {
        myResult = SEARCH_NOT_FOUND;
        to_be_searched = false;
      }
    } else {
      // either we are at a leaf node and have found a match
      // or, we are at an intermediate node and should go the next level
      next = sycl::select_from_group(sg, src_unit_data, dest_lane_pivot - 1);
      found = (isLeaf && src_unit_data == src_key);
      found = sycl::select_from_group(sg, found, dest_lane_pivot);

      if (found && (laneId == src_lane)) {  // leaf and found
        myResult = next;
        to_be_searched = false;
      }

      if (isLeaf && !found && (laneId == src_lane)) {  // leaf and not found
        myResult = SEARCH_NOT_FOUND;
        to_be_searched = false;
      }
    }
    last_work_queue = work_queue;
  }
}

#define KEY_PIVOT_MASK_R 0x15555555

template <typename KeyT>
void delete_unit_bulk(uint32_t &laneId, KeyT &myKey,
                      uint32_t *d_root, uint32_t *d_pool,
                      sycl::nd_item<1> &item) {
  int dest_lane_pivot;
  uint32_t rootAddress = *d_root;
  auto sg = item.get_sub_group();

#pragma unroll
  for (int src_lane = 0; src_lane < WARP_WIDTH; src_lane++) {
    KeyT src_key = sycl::select_from_group(sg, myKey, src_lane);
    KeyT next = rootAddress;
    bool isIntermediate = true;
    do {
      KeyT src_unit_data = *(getAddressPtr(next) + laneId);
      isIntermediate = !((src_unit_data & 0x80000000) == 0);  // only even lanes are valid
      isIntermediate = sycl::select_from_group(sg, isIntermediate, 0);
      if (!isIntermediate) {
        acquire_lock(getAddressPtr(next), item);
        src_unit_data = volatileNodeRead(getAddressPtr(next), item);
      }
      KeyT src_unit_key = src_unit_data & 0x7FFFFFFF;
      bool hit = (src_key >= src_unit_key) && src_unit_key;
      bool key_exist = sycl::reduce_over_group(
          sg, (src_key == src_unit_key) & KEY_PIVOT_MASK_R
              ? (0x1 << sg.get_local_linear_id()) : 0,
          sycl::ext::oneapi::plus<>());
             
      uint32_t isFoundPivot_bmp = sycl::reduce_over_group(
          sg, hit ? (0x1 << sg.get_local_linear_id()) : 0,
          sycl::ext::oneapi::plus<>());
      dest_lane_pivot = __ffs(~isFoundPivot_bmp & KEY_PIVOT_MASK_R);
      if (isIntermediate) {
        dest_lane_pivot = dest_lane_pivot ? dest_lane_pivot - 2 : 29;
        next = sycl::select_from_group(sg, src_unit_data, dest_lane_pivot);
      } else {
        if (key_exist) {
          uint32_t newNodeData = sycl::shift_group_left(sg, src_unit_key, 2);
          isFoundPivot_bmp &= KEY_PIVOT_MASK_R;
          isFoundPivot_bmp |= (isFoundPivot_bmp << 1);  // mark values
          isFoundPivot_bmp >>= 2;                       // remove mask for src_key
          bool to_move = ((1 << laneId) & ~isFoundPivot_bmp) && (laneId < 30);
          KeyT finalData = (to_move * newNodeData + (!to_move) * src_unit_key);
          finalData = (laneId >= 28 && laneId < 30) ? 0 : finalData;
          finalData = (laneId == 1) ? finalData | 0x80000000 : finalData;
          volatileNodeWrite(getAddressPtr(next), finalData, item);
        }
        release_lock(getAddressPtr(next), item);
      }
    } while (isIntermediate);
  }
}

template <typename KeyT, typename ValueT, typename SizeT>
void range_unit(uint32_t &laneId, bool &to_search,
                KeyT &lower_bound, KeyT &upper_bound,
                ValueT *range_results, uint32_t *d_root,
                SizeT &range_length, uint32_t *d_pool,
                sycl::nd_item<1> &item) {
  int dest_lane_pivot;
  uint32_t rootAddress = *d_root;
  auto sg = item.get_sub_group();

  while (auto work_queue = sycl::reduce_over_group(
             sg, to_search ? (0x1 << sg.get_local_linear_id()) : 0,
             sycl::ext::oneapi::plus<>())) {
    auto src_lane = __ffs(work_queue) - 1;
    KeyT src_key_lower = sycl::select_from_group(sg, lower_bound, src_lane);
    KeyT src_key_upper = sycl::select_from_group(sg, upper_bound, src_lane);
    KeyT next = rootAddress;
    bool is_intermediate = true;
    if (laneId == src_lane)
      to_search = false;
    do {
      uint32_t src_unit_data = *(getAddressPtr(next) + laneId);
      is_intermediate = !((src_unit_data & 0x80000000) == 0);
      is_intermediate = sycl::select_from_group(sg, is_intermediate, 0);

      uint32_t src_unit_key = src_unit_data & 0x7FFFFFFF;
      bool hit = (src_key_lower >= src_unit_key) && src_unit_key;
      uint32_t isFoundPivot_bmp = sycl::reduce_over_group(
          sg, hit ? (0x1 << sg.get_local_linear_id()) : 0,
          sycl::ext::oneapi::plus<>());
      dest_lane_pivot = __ffs(~isFoundPivot_bmp & KEY_PIVOT_MASK_R);
      if (is_intermediate) {
        dest_lane_pivot = dest_lane_pivot ? dest_lane_pivot - 2 : 29;
        next = sycl::select_from_group(sg, src_unit_data, dest_lane_pivot);
      } else {
        uint32_t tid = item.get_global_id(0);
        tid /= 32;
        tid *= 32;
        tid += src_lane;
        uint32_t offset = tid * range_length * 2;
        while (true) {
          hit = ((src_key_lower <= src_unit_key && src_key_upper >= src_unit_key) &&
                 src_unit_key);
          isFoundPivot_bmp = sycl::reduce_over_group(
              sg, hit ? (0x1 << sg.get_local_linear_id()) : 0,
              sycl::ext::oneapi::plus<>());
          isFoundPivot_bmp &= KEY_PIVOT_MASK_R;
          dest_lane_pivot = __ffs(isFoundPivot_bmp);

          dest_lane_pivot--;
          isFoundPivot_bmp >>= dest_lane_pivot;
          uint32_t link_min =
              sycl::select_from_group(sg, src_unit_key, 30) & 0x7FFFFFFF; // get link min
          src_unit_key = sycl::shift_group_left(sg, src_unit_key, dest_lane_pivot);
          uint32_t counter = sycl::popcount(isFoundPivot_bmp) * 2;
          if (laneId < counter)
            range_results[offset + laneId] = src_unit_key - 2;
          if (!link_min || src_key_upper < link_min)
            break;            // done
          offset += counter;  // load next node
          next = sycl::select_from_group(sg, src_unit_key, 31) & 0x7FFFFFFF;
          src_unit_key = *(getAddressPtr(next) + laneId);
        }
      }
    } while (is_intermediate);
  }
}
}  // namespace warps
}  // namespace GpuBTree
