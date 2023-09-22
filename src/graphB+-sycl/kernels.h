static const int ThreadsPerBlock = 256;
static const int warpsize = 32;

inline int atomicCAS(int *val, int expected, int desired)
{
  int expected_value = expected;
  auto atm = sycl::atomic_ref<int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(*val);
  atm.compare_exchange_strong(expected_value, desired);
  return expected_value;
}

inline int atomicAdd(int *val, int operand)
{
  auto atm = sycl::atomic_ref<int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(*val);
  return atm.fetch_add(operand);
}

inline int atomicAnd(int *val, int operand)
{
  auto atm = sycl::atomic_ref<int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(*val);
  return atm.fetch_and(operand);
}

inline void atomicMax(unsigned long long *val, unsigned long long operand)
{
  auto atm = sycl::atomic_ref<unsigned long long,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(*val);
  atm.fetch_max(operand);
}

inline int __ffs(int x) {
  return (x == 0) ? 0 : sycl::ctz(x) + 1;
}

template <typename T>
static 
void swap(T& a, T& b)
{
  T c = a;
  a = b;
  b = c;
}

static 
int representative(const int idx, int* const __restrict label)
{
  int curr = label[idx];
  if (curr != idx) {
    int next, prev = idx;
    while (curr > (next = label[curr])) {
      label[prev] = next;
      prev = curr;
      curr = next;
    }
  }
  return curr;
}

// source of hash function: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static 
unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

struct EdgeInfo {
  int beg;  // beginning of range (shifted by 1) | is range inverted or not
  int end;  // end of range (shifted by 1) | plus or minus (1 = minus, 0 = plus or zero)
};

struct Graph {
  int nodes;
  int edges;
  int* nindex;  // first CSR array
  int* nlist;  // second CSR array
  int* eweight;  // edge weights (-1, 0, 1)
  int* origID;  // original node IDs
};

static Graph readGraph(const char* const name)
{
  // read input from file
  FILE* fin = fopen(name, "rt");
  if (fin == NULL) {printf("ERROR: could not open input file %s\n", name); exit(-1);}
  size_t linesize = 256;
  char buf[linesize];
  char* ptr = buf;
  getline(&ptr, &linesize, fin);  // skip first line

  int selfedges = 0, wrongweights = 0, duplicates = 0, inconsistent = 0, line = 1, cnt = 0;
  int src, dst, wei;
  std::map<int, int> map;  // map node IDs to contiguous IDs
  std::set<std::pair<int, int>> set2;
  std::set<std::tuple<int, int, int>> set3;
  while (fscanf(fin, "%d,%d,%d", &src, &dst, &wei) == 3) {
    if (src == dst) {
      selfedges++;
    } else if ((wei < -1) || (wei > 1)) {
      wrongweights++;
    } else if (set2.find(std::make_pair(std::min(src, dst), std::max(src, dst))) != set2.end()) {
      if (set3.find(std::make_tuple(std::min(src, dst), std::max(src, dst), wei)) != set3.end()) {
        duplicates++;
      } else {
        inconsistent++;
      }
    } else {
      set2.insert(std::make_pair(std::min(src, dst), std::max(src, dst)));
      set3.insert(std::make_tuple(std::min(src, dst), std::max(src, dst), wei));
      if (map.find(src) == map.end()) {
        map[src] = cnt++;
      }
      if (map.find(dst) == map.end()) {
        map[dst] = cnt++;
      }
    }
    line++;
  }
  fclose(fin);

  // print stats
  printf("  read %d lines\n", line);
  if (selfedges > 0) printf("  skipped %d self-edges\n", selfedges);
  if (wrongweights > 0) printf("  skipped %d edges with out-of-range weights\n", wrongweights);
  if (duplicates > 0) printf("  skipped %d duplicate edges\n", duplicates);
  if (inconsistent > 0) printf("  skipped %d inconsistent edges\n", inconsistent);
  if ((int)map.size() != cnt) printf("ERROR: wrong node count\n");
  printf("  number of unique nodes: %d\n", (int)map.size());
  printf("  number of unique edges: %d\n", (int)set3.size());

  // compute CCs with union find
  int* const label = new int [cnt];
  for (int v = 0; v < cnt; v++) {
    label[v] = v;
  }
  for (auto ele: set3) {
    const int src = map[std::get<0>(ele)];
    const int dst = map[std::get<1>(ele)];
    const int vstat = representative(src, label);
    const int ostat = representative(dst, label);
    if (vstat != ostat) {
      if (vstat < ostat) {
        label[ostat] = vstat;
      } else {
        label[vstat] = ostat;
      }
    }
  }
  for (int v = 0; v < cnt; v++) {
    int next, vstat = label[v];
    while (vstat > (next = label[vstat])) {
      vstat = next;
    }
    label[v] = vstat;
  }

  // determine CC sizes
  int* const size = new int [cnt];
  for (int v = 0; v < cnt; v++) {
    size[v] = 0;
  }
  for (int v = 0; v < cnt; v++) {
    size[label[v]]++;
  }

  // find largest CC
  int hi = 0;
  for (int v = 1; v < cnt; v++) {
    if (size[hi] < size[v]) hi = v;
  }

  // keep if in largest CC and convert graph into set format
  Graph g;
  g.origID = new int [cnt];  // upper bound on size
  int nodes = 0, edges = 0;
  std::map<int, int> newmap;  // map node IDs to contiguous IDs
  std::set<std::pair<int, int>>* const node = new std::set<std::pair<int, int>> [cnt];  // upper bound on size
  for (auto ele: set3) {
    const int src = std::get<0>(ele);
    const int dst = std::get<1>(ele);
    const int wei = std::get<2>(ele);
    if (label[map[src]] == hi) {  // in largest CC
      if (newmap.find(src) == newmap.end()) {
        g.origID[nodes] = src;
        newmap[src] = nodes++;
      }
      if (newmap.find(dst) == newmap.end()) {
        g.origID[nodes] = dst;
        newmap[dst] = nodes++;
      }
      node[newmap[src]].insert(std::make_pair(newmap[dst], wei));
      node[newmap[dst]].insert(std::make_pair(newmap[src], wei));
      edges += 2;
    }
  }

  if (nodes > cnt) printf("ERROR: too many nodes\n");
  if (edges > (int)set3.size() * 2) printf("ERROR: too many edges\n");

  // create graph in CSR format
  g.nodes = nodes;
  g.edges = edges;
  g.nindex = new int [g.nodes + 1];
  g.nlist = new int [g.edges];
  g.eweight = new int [g.edges];
  int acc = 0;
  for (int v = 0; v < g.nodes; v++) {
    g.nindex[v] = acc;
    for (auto ele: node[v]) {
      const int dst = ele.first;
      const int wei = ele.second;
      g.nlist[acc] = dst;
      g.eweight[acc] = wei;
      acc++;
    }
  }
  g.nindex[g.nodes] = acc;

  if (acc != edges) printf("ERROR: wrong edge count in final graph\n");

  delete [] label;
  delete [] size;
  delete [] node;

  return g;
}

static void freeGraph(Graph &g)
{
  g.nodes = 0;
  g.edges = 0;
  delete [] g.nindex;
  delete [] g.nlist;
  delete [] g.eweight;
  delete [] g.origID;
  g.nindex = NULL;
  g.nlist = NULL;
  g.eweight = NULL;
  g.origID = NULL;
}

static void init(
  const int edges,
  const int nodes, 
  int* const nlist,
  int* const eweight,
  int* const inCC,
  EdgeInfo* const einfo,
  int* const inTree,
  int* const negCnt,
  sycl::nd_item<1> &item)
{
  const int from = item.get_global_id(0);
  const int incr = item.get_group_range(0) * ThreadsPerBlock;

  for (int j = from; j < edges; j += incr) {
    nlist[j] <<= 1;
  }

  // zero out inCC
  for (int v = from; v < nodes; v += incr) {
    inCC[v] = 0;
  }

  // set minus if graph weight is -1
  for (int j = from; j < edges; j += incr) {
    einfo[j].end = (eweight[j] == -1) ? 1 : 0;
  }

  // zero out inTree and negCnt
  for (int j = from; j < edges; j += incr) {
    inTree[j] = 0;
    negCnt[j] = 0;
  }
}

static void init2(
  const int edges,
  const int nodes,
  const int root,
  int* const nlist,
  int* const parent,
  int* const queue,
  int* const label,
  int* const tail,
  sycl::nd_item<1> &item)
{
  const int from = item.get_global_id(0);
  const int incr = item.get_group_range(0) * ThreadsPerBlock;
  // initialize
  for (int j = from; j < edges; j += incr) nlist[j] &= ~1;
  for (int j = from; j < nodes; j += incr) parent[j] = (root == j) ? (INT_MAX & ~3) : -1;
  for (int j = from; j < nodes; j += incr) label[j] = 1;
  if (from == 0) {
    queue[0] = root;
    *tail = 1;
  }
}

static void generateSpanningTree(
  const int nodes,
  const int* const __restrict nindex,
  const int* const __restrict nlist,
  const int seed,
  EdgeInfo* const einfo,
  volatile int* const parent,
  int* const queue,
  const int level,
  int* const tail,
  int start,
  int end,
  sycl::nd_item<1> &item)
{
  auto sg = item.get_sub_group();
  const int from = item.get_global_id(0) / warpsize;
  const int incr = item.get_group_range(0) * ThreadsPerBlock / warpsize;
  const int lane = item.get_local_id(0) % warpsize;
  const int seed2 = seed * seed + seed;
  const int bit = (level & 1) | 2;

  for (int i = start + from; i < end; i += incr) {
    const int node = queue[i];
    const int me = (node << 2) | bit;
    if (lane == 0)
        atomicAnd((int *)(parent+node), ~3);
    for (int j = nindex[node + 1] - 1 - lane; j >= nindex[node]; j -= warpsize) {  // reverse order on purpose
      const int neighbor = nlist[j] >> 1;
      const int seed3 = neighbor ^ seed2;
      const int hash_me = hash(me ^ seed3);
      int val, hash_val;
      do {  // pick parent deterministically
          val = parent[neighbor];
          hash_val = hash(val ^ seed3);
        } while (((val < 0) || (((val & 3) == bit) && ((hash_val < hash_me) ||
                 ((hash_val == hash_me) && (val < me))))) &&
                 (atomicCAS((int*)(parent+neighbor), val, me) != val));
        if (val < 0) {
          val = atomicAdd(tail, 1);
          queue[val] = neighbor;
      }
    }
    sycl::group_barrier(sg);
  }
}

#ifdef VERIFY
static void verify_generateSpanningTree(
  const int nodes,
  const int edges,
  const int* const nindex,
  const int* const nlist,
  const int seed,
  const int* const parent,
  const int level,
  const int* const tail,
        int end,
        sycl::nd_item<1> &item,
        const sycl::stream &stream_ct1)
{
  const int from = item.get_global_id(0);
  const int incr = item.get_group_range(0) * ThreadsPerBlock;
  if (end != *tail) stream_ct1 << "ERROR: head mismatch\n";
  if (*tail != nodes) {
    stream_ct1 << "ERROR: tail mismatch tail " << *tail << " nodes " << nodes << "\n";
  }
  for (int i = from; i < nodes; i += incr) {
    if (parent[i] < 0) {
      stream_ct1 << "ERROR: found unvisited node " << i << "\n";
    }
  }
}
#endif

static void rootcount(
  const int* const parent,
  const int* const queue,
        int* const __restrict label,
  const int level,
        int start,
        int end,
        sycl::nd_item<1> &item)
{
  const int from = item.get_global_id(0);
  const int incr = item.get_group_range(0) * ThreadsPerBlock;
  // bottom up: push counts
  for (int i = start + from; i < end; i += incr) {
    const int node = queue[i];
    atomicAdd(&label[parent[node] >> 2], label[node]);
  }
}

static void treelabel(
  const int nodes,
  const int* const __restrict nindex,
  volatile int* const __restrict nlist,
  EdgeInfo* const __restrict einfo,
  volatile int* const __restrict inTree,
  volatile int* const __restrict negCnt,
  const int* const __restrict parent,
  const int* const __restrict queue,
        int* const __restrict label,
  const int level,
        int start,
        int end,
        sycl::nd_item<1> &item
#ifdef VERIFY
        , const sycl::stream &stream_ct1
#endif
        )
{
  auto sg = item.get_sub_group();
  const int from = item.get_global_id(0) / warpsize;
  const int incr = item.get_group_range(0) * ThreadsPerBlock / warpsize;
  const int lane = item.get_local_id(0) % warpsize;

  //cuv
  // top down: label tree + set nlist flag + set edge info + move tree nodes to front + make parent edge first in list
  for (int i = start + from; i < end; i += incr) {
    const int node = queue[i];
    const int par = parent[node] >> 2;
    const int nodelabel = label[node];
    const int beg = nindex[node];
    const int end = nindex[node + 1];

    // set nlist flag + set edge info
    int lbl = (nodelabel >> 1) + 1;
    for (int j = beg + lane; sycl::any_of_group(sg, j < end); j += warpsize) {
      int lblinc = 0;
      int neighbor = -1;
      bool cond = false;
      if (j < end) {
        neighbor = nlist[j] >> 1;
        cond = (neighbor != par) && ((parent[neighbor] >> 2) == node);
        if (cond) {
          lblinc = label[neighbor];
        }
      }
      const int currcount = lblinc;
      for (int d = 1; d < 32; d *= 2) {
        const int tmp =
            sycl::shift_group_right(sg, lblinc, d);
        if (lane >= d) lblinc += tmp;
      }
      lbl += lblinc;

      if (cond) {
        const int lblval = (lbl - currcount) << 1;
        label[neighbor] = lblval;
        einfo[j].beg = lblval;
        einfo[j].end = (einfo[j].end & 1) | ((lbl - 1) << 1);
        nlist[j] |= 1;  // child edge is in tree
      }
      lbl = sycl::select_from_group(sg, lbl, 31);
    }

    // move tree nodes to front
    const int len = end - beg;
    if (len > 0) {
      enum {none, some, left, right};
      if (len <= warpsize) {
        const int src = beg + lane;
        int b, e, in, neg,  n, state = none;
        if (lane < len) {
          b = einfo[src].beg;
          e = einfo[src].end;
          in = inTree[src];
          neg = negCnt[src];
          n = nlist[src];
          const int neighbor = n >> 1;
          state = ((neighbor != par) && ((parent[neighbor] >> 2) == node)) ? left : right;  // partitioning condition
        }
        const int ball = sycl::reduce_over_group(
            sg, state == left ? (0x1 << sg.get_local_linear_id()) : 0,
            sycl::ext::oneapi::plus<>());
        const int balr = sycl::reduce_over_group(
            sg, state == right ? (0x1 << sg.get_local_linear_id()) : 0,
            sycl::ext::oneapi::plus<>());
        const int pfsl = sycl::popcount(ball & ~(-1 << lane));
        const int pfsr = sycl::popcount(balr & ~(-1 << lane));
        const int pos = beg + ((state == right) ? (len - 1 - pfsr) : pfsl);
        if (state != none) {
          einfo[pos].beg = b;
          einfo[pos].end = e;
          inTree[pos] = in;
          negCnt[pos] = neg;
          nlist[pos] = n;
        }
      } else {
        int lp = beg;
        int rp = end - 1;
        int state = some;
        int read = beg + sycl::min(warpsize, len);
        int src = beg + lane;
        int b = einfo[src].beg;
        int e = einfo[src].end;
        int n = nlist[src];
        int in = inTree[src];
        int neg = negCnt[src];

        do {
          if (state == some) {
            const int neighbor = n >> 1;
            state = ((neighbor != par) && ((parent[neighbor] >> 2) == node)) ? left : right;  // partitioning condition
          }
          const int ball = sycl::reduce_over_group(
              sg, state == left ? (0x1 << sg.get_local_linear_id()) : 0,
              sycl::ext::oneapi::plus<>());
          const int pfsl = sycl::popcount(ball & ~(-1 << lane));
          if (state == left) {
            int oldb, olde, oldin, oldneg, oldn;
            const int pos = lp + pfsl;
            if (pos >= read) {
              oldb = einfo[pos].beg;
              olde = einfo[pos].end;
              oldin = inTree[pos];
              oldneg = negCnt[pos];
              oldn = nlist[pos];
            }
            einfo[pos].beg = b;
            einfo[pos].end = e;
            inTree[pos] = in;
            negCnt[pos] = neg;
            nlist[pos] = n;
            b = oldb;
            e = olde;
            in = oldin;
            neg = oldneg;
            n = oldn;
            state = (pos < read) ? none : some;
          }
          lp += sycl::popcount(ball);
          read = sycl::max(read, lp);
          const int balr = sycl::reduce_over_group(
              sg, state == right ? (0x1 << sg.get_local_linear_id()) : 0,
              sycl::ext::oneapi::plus<>());
          const int pfsr = sycl::popcount(balr & ~(-1 << lane));
          if (state == right) {
            int oldb, olde, oldin, oldneg, oldn;
            const int pos = rp - pfsr;
            if (pos >= read) {
              oldb = einfo[pos].beg;
              olde = einfo[pos].end;
              oldin = inTree[pos];
              oldneg = negCnt[pos];
              oldn = nlist[pos];
            }
            einfo[pos].beg = b;
            einfo[pos].end = e;
            inTree[pos] = in;
            negCnt[pos] = neg;
            nlist[pos] = n;
            b = oldb;
            e = olde;
            in = oldin;
            neg = oldneg;
            n = oldn;
            state = (pos < read) ? none : some;
          }
          rp -= sycl::popcount(balr);
          if (read <= rp) {
            const int bal = sycl::reduce_over_group(
                sg, state == none ? (0x1 << sg.get_local_linear_id()) : 0,
                sycl::ext::oneapi::plus<>());
            const int pfs = sycl::popcount(bal & ~(-1 << lane));
            if (state == none) {
              const int pos = read + pfs;
              if (pos <= rp) {
                b = einfo[pos].beg;
                e = einfo[pos].end;
                in = inTree[pos];
                neg = negCnt[pos];
                n = nlist[pos];
                state = some;
              }
            }
            read += sycl::popcount(bal); // may be too high but okay
          }
        } while (sycl::any_of_group(sg, state == some));
      }
    }

    //find paredge here
    int paredge = -1;
    for (int j = beg + lane; sycl::any_of_group(sg, j < end); j += warpsize) {
      if (j < end) {
        const int neighbor = nlist[j] >> 1;
        if (neighbor == par) {
          paredge = j;
        }
      }
      if (sycl::any_of_group(sg, paredge >= 0)) break;
    }
    int pos = -1;
    for (int j = beg + lane; sycl::any_of_group(sg, j < end); j += warpsize) {
      if (j < end) {
        const int neighbor = nlist[j] >> 1;
        if (((parent[neighbor] >> 2) != node)) {
          pos = j;
        }
      }
      if (sycl::any_of_group(sg, pos >= 0)) break;
    }
    unsigned int bal = sycl::reduce_over_group(
        sg, pos >= 0 ? (0x1 << sg.get_local_linear_id()) : 0,
        sycl::ext::oneapi::plus<>());

    const int lid = __ffs(bal) - 1;
    pos = sycl::select_from_group(sg, pos, lid);
    if (paredge >= 0) {  // only one thread per warp
      einfo[paredge].beg = nodelabel | 1;
      einfo[paredge].end = (einfo[paredge].end & 1) | ((lbl - 1) << 1);
      nlist[paredge] |= 1;
      if (paredge != beg) {
        if (paredge != pos) {
          swap(nlist[pos], nlist[paredge]);
          swap(einfo[pos], einfo[paredge]);
          swap(inTree[pos], inTree[paredge]);
          swap(negCnt[pos], negCnt[paredge]);
          paredge = pos;
        }
        if (paredge != beg) {
          swap(nlist[beg], nlist[paredge]);
          swap(einfo[beg], einfo[paredge]);
          swap(inTree[beg], inTree[paredge]);
          swap(negCnt[beg], negCnt[paredge]);
        }
      }
    }
    sycl::group_barrier(sg);

#ifdef VERIFY
    if (lane == 0) {
      if (i == 0) {
        if (lbl != nodes) {
          stream_ct1 << "ERROR: lbl mismatch, lbl " << lbl << " nodes " << nodes << "\n";
        }
      }
      int j = beg;
      while ((j < end) && (nlist[j] & 1)) j++;
      while ((j < end) && !(nlist[j] & 1)) j++;
      if (j != end) {
        stream_ct1 << "ERROR: not moved " << beg << " " << j << " " << end;
      }
    }
#endif
  }
}

static void inTreeUpdate(
  const int edges,
  const int* const __restrict nlist,
  volatile int* const __restrict inTree,
  sycl::nd_item<1> &item)
{
  const int from = item.get_global_id(0);
  const int incr = item.get_group_range(0) * ThreadsPerBlock;
  // update inTree
  for (int j = from; j < edges; j += incr) {
    inTree[j] += nlist[j] & 1;
  }
}

static void processCycles(
  const int nodes,
  const int* const __restrict nindex,
  const int* const __restrict nlist,
  const int* const __restrict label,
  const EdgeInfo* const __restrict einfo,
  bool* const  __restrict minus,
  sycl::nd_item<1> &item
#ifdef VERIFY
  , const sycl::stream &stream_ct1
#endif
  )
{
  auto sg = item.get_sub_group();
  const int from = item.get_global_id(0) / warpsize;
  const int incr = item.get_group_range(0) * ThreadsPerBlock / warpsize;
  const int lane = item.get_local_id(0) % warpsize;

  for (int i = from; i < nodes; i += incr) {
    const int target0 = label[i];
    const int target1 = target0 | 1;
    int j = nindex[i + 1] - 1 - lane;
    while ((j >= nindex[i]) && !(nlist[j] & 1)) {
      int curr = nlist[j] >> 1;
      if (curr > i) {  // only process edges in one direction
        int sum = 0;
        while (label[curr] != target0) {
          int k = nindex[curr];
          while ((einfo[k].beg & 1) == ((einfo[k].beg <= target1) && (target0 <= einfo[k].end))) k++;
#ifdef VERIFY
          if ((k >= nindex[curr + 1]) || !(nlist[k] & 1)) {
            stream_ct1 << "ERROR: couldn't find path\n";
          }
#endif
          sum += einfo[k].end & 1;
          curr = nlist[k] >> 1;
        }
        minus[j] = sum & 1;
      }
      j -= warpsize;
    }
    sycl::group_barrier(sg);
  }
}

static void initMinus(
  const int edges,
  const int nodes,
  const int* const __restrict nindex,
  const int* const __restrict nlist,
  const EdgeInfo* const einfo,
  bool* const minus,
  sycl::nd_item<1> &item)
{
  const int from = item.get_global_id(0);
  const int incr = item.get_group_range(0) * ThreadsPerBlock;
  // set minus info to true
  for (int j = from; j < edges; j += incr) {
    minus[j] = true;
  }
  // copy minus info of tree edges
  for (int i = from; i < nodes; i += incr) {
    int j = nindex[i];
    while ((j < nindex[i + 1]) && (nlist[j] & 1)) {
      minus[j] = einfo[j].end & 1;
      j++;
    }
  }
}

static void init3(
  const int nodes,
  const int* const __restrict nidx,
  const int* const __restrict nlist,
  int* const __restrict label,
  int* const __restrict count,
  sycl::nd_item<1> &item)
{
  const int from = item.get_global_id(0);
  const int incr = item.get_group_range(0) * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr) {
    label[v] = v;
  }

  for (int v = from; v < nodes; v += incr) {
    count[v] = 0;
  }
}

static void compute1(
  const int nodes,
  const int* const __restrict nidx,
  const int* const __restrict nlist,
  int* const __restrict label,
  const bool* const __restrict minus,
  int* const __restrict negCnt,
  sycl::nd_item<1> &item)
{
  const int from = item.get_global_id(0) / warpsize;
  const int incr = item.get_group_range(0) * ThreadsPerBlock / warpsize;
  const int lane = item.get_local_id(0) % warpsize;
  for (int v = from; v < nodes; v += incr) {
    const int beg = nidx[v];
    const int end = nidx[v + 1];
    int vstat = representative(v, label);
    for (int j = beg + lane; j < end; j += warpsize) {
      const int nli = nlist[j] >> 1;
      if (minus[j]) {
        negCnt[j]++;
      } else {
        int ostat = representative(nli, label);
        bool repeat;
        do {
          repeat = false;
          if (vstat != ostat) {
            int ret;
            if (vstat < ostat) {
              if ((ret = atomicCAS(label+ostat, ostat, vstat)) != ostat) {
                ostat = ret;
                repeat = true;
              }
            } else {
              if ((ret = atomicCAS(label+vstat, vstat, ostat)) != vstat) {
                vstat = ret;
                repeat = true;
              }
            }
          }
        } while (repeat);
      }
    }
  }
}

static void flatten(const int nodes, int* const __restrict label,
                    sycl::nd_item<1> &item)
{
  const int from = item.get_global_id(0);
  const int incr = item.get_group_range(0) * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr) {
    int next, vstat = label[v];
    const int old = vstat;
    while (vstat > (next = label[vstat])) {
      vstat = next;
    }
    if (old != vstat) label[v] = vstat;
  }
}

static void ccSize(
  const int nodes,
  const int* const __restrict label,
        int* const __restrict count,
        sycl::nd_item<1> item,
        unsigned long long *hi,
        int *wSize)
{
  const int from = item.get_global_id(0);
  const int incr = item.get_group_range(0) * ThreadsPerBlock;
  if (from == 0)
  {
    *hi = 0;
    *wSize = 0;
  }
  for (int v = from; v < nodes; v += incr) {
    atomicAdd(&count[label[v]], 1);
  }
}

static void largestCC(const int nodes, const int* const __restrict count,
                      sycl::nd_item<1> &item, unsigned long long *hi)
{
  const int from = item.get_global_id(0);
  const int incr = item.get_group_range(0) * ThreadsPerBlock;
  for (int v = from; v < nodes; v += incr) {
    const unsigned long long d_hi = (((unsigned long long)count[v]) << 32)| v;
    if (*hi < d_hi) {
      atomicMax(hi, d_hi);
    }
  }
}

static void ccHopCount(
  const int nodes,
  const int* const __restrict nidx,
  const int* const __restrict nlist,
  const int* const __restrict label,
  int* const __restrict count,
  int* const __restrict ws1,
  int* const __restrict ws2,
  sycl::nd_item<1> &item,
  unsigned long long *hi,
  int *wSize)
{
  const int from = item.get_global_id(0) / warpsize;
  const int incr = (item.get_group_range(0) * ThreadsPerBlock) / warpsize;
  const int lane = item.get_local_id(0) % warpsize;

  const int hi2 = *hi & 0xffffffff;
  for (int v = from; v < nodes; v += incr) {
    const int lblv = label[v];
    if (lblv == v) {
      count[lblv] = (lblv == hi2) ? 0 : INT_MAX - 1;  // init count
    }
    for (int j = nidx[v] + lane; j < nidx[v + 1]; j += warpsize) {
      const int nli = nlist[j] >> 1;
      const int lbln = label[nli];
      if (lblv < lbln) {  // only one direction
        const int idx = atomicAdd(wSize, 1);  // get the return value and use it
        ws1[idx] = lblv;
        ws2[idx] = lbln;
      }
    }
  }
}

static void BellmanFord(
  int* const __restrict count,
  bool* const __restrict changed,
  const int* const __restrict ws1,
  const int* const __restrict ws2,
  sycl::nd_item<1> &item,
  int *wSize)
{
  const int from = item.get_global_id(0);
  const int incr = item.get_group_range(0) * ThreadsPerBlock;
  // use Bellman Ford to compute distances
  for (int i = from; i < *wSize; i += incr) {
    const int lblv = ws1[i];
    const int lbln = ws2[i];
    const int distv = count[lblv];
    const int distn = count[lbln];
    if (distv + 1 < distn) {
      count[lbln] = distv + 1;
      *changed = true;
    } else if (distn + 1 < distv) {
      count[lblv] = distn + 1;
      *changed = true;
    }
  }
}

static void incrementCC(
  const int nodes,
  const int* const __restrict label,
  const int* const __restrict count,
  int* const __restrict inCC,
  sycl::nd_item<1> &item)
{
  const int from = item.get_global_id(0);
  const int incr = item.get_group_range(0) * ThreadsPerBlock;
  // increment inCC if node is at even hop count from source CC
  for (int v = from; v < nodes; v += incr) {
    inCC[v] += (count[label[v]] % 2) ^ 1;
  }
}
