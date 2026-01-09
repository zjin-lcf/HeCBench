# Missing Data Files - Comprehensive Fix Plan
**Date:** January 8, 2026
**Affected Benchmarks:** 21

## Summary

21 benchmarks need data files to run. Total of ~31 unique missing data files identified.

## Categories of Missing Data

### 1. Image Processing Data (7 benchmarks)
**Easy to generate** - can create synthetic test images

| Benchmark | Missing Files | Solution |
|-----------|---------------|----------|
| boxfilter-omp | data/lenaRGB.ppm | Generate or download Lena test image |
| dxtc2-omp | data/lena_std.ppm, data/lena_ref.dds<br>data/teapot512_std.ppm, data/teapot512_ref.dds | Generate test images |
| medianfilter-omp | data/SierrasRGB.ppm | Generate or download Sierra test image |
| sobel-omp | ../sobel-sycl/SobelFilter_Input.bmp | Generate test bitmap |
| face-omp | ../face-cuda/info.txt<br>../face-cuda/class.txt | Copy from CUDA or generate |

### 2. Graph/Network Data (5 benchmarks)
**Medium difficulty** - need to generate or download graph structures

| Benchmark | Missing Files | Solution |
|-----------|---------------|----------|
| bfs-omp | ../data/bfs/graph1MW_6.txt | Generate synthetic graph (1M vertices) |
| b+tree-omp | ../data/b+tree/mil.txt<br>../data/b+tree/command.txt | Generate B-tree test data |
| lanczos-omp | data/social-large-800k.txt | Generate social network graph |
| kmeans-omp | ../data/kmeans/kdd_cup | Download KDD Cup dataset or generate |

### 3. Scientific Computing Data (4 benchmarks)
**Easy to generate** - random/synthetic numerical data

| Benchmark | Missing Files | Solution |
|-----------|---------------|----------|
| hotspot3D-omp | ../data/hotspot3D/power_512x8<br>../data/hotspot3D/temp_512x8 | Generate 512x8 matrices (binary) |
| cfd-omp | ../data/cfd/fvcorr.domn.097K<br>../data/cfd/fvcorr.domn.193K | Generate CFD domain files |
| svd3x3-omp | ../svd3x3-cuda/Dataset_1M.txt | Generate 1M 3x3 matrices |
| permutate-omp | data/truerand_1bit.bin | Generate random binary data |

### 4. Bioinformatics Data (3 benchmarks)
**Medium difficulty** - need appropriate format

| Benchmark | Missing Files | Solution |
|-----------|---------------|----------|
| diamond-omp | $(DMND_PATH)/long.fastq.gz | Generate synthetic FASTQ sequences |
| leukocyte-omp | ../data/leukocyte/testfile.avi | Generate test video or use sample |
| minimap2-omp | ../minimap2-sycl/in-1k.txt | Generate synthetic DNA sequences |

### 5. Other Data (2 benchmarks)

| Benchmark | Missing Files | Solution |
|-----------|---------------|----------|
| ccs-omp | ../ccs-cuda/Data_Constant_100_1_bicluster.txt | Generate bicluster data |
| cmp-omp | data/simple-synthetic.su | Generate seismic data |
| hogbom-omp | data/dirty_4096.img<br>data/psf_4096.img | Generate radio astronomy images |
| minibude-omp | data/bm1 | Copy from miniBUDE repo or generate |
| ss-omp | ../ss-sycl/StringSearch_Input.txt | Generate string search input |

---

## Fix Priority

### Priority 1: Easy Synthetic Data (12 benchmarks)
**Can fix quickly** - simple data generation scripts

1. **hotspot3D-omp** - Generate random 512x8 matrices
2. **svd3x3-omp** - Generate random 3x3 matrices
3. **permutate-omp** - Generate random binary file
4. **boxfilter-omp** - Generate test image
5. **medianfilter-omp** - Generate test image
6. **sobel-omp** - Generate test bitmap
7. **dxtc2-omp** - Generate test images
8. **ss-omp** - Generate test strings
9. **ccs-omp** - Generate bicluster data
10. **face-omp** - Generate classifier data
11. **hogbom-omp** - Generate radio images
12. **cmp-omp** - Generate seismic traces

### Priority 2: Graph Data (4 benchmarks)
**Needs graph generation** - slightly more complex

1. **bfs-omp** - Generate 1M node graph
2. **b+tree-omp** - Generate B-tree commands
3. **lanczos-omp** - Generate social network
4. **kmeans-omp** - Generate clustering data or download KDD

### Priority 3: Bioinformatics (3 benchmarks)
**Needs domain knowledge** - proper format required

1. **diamond-omp** - FASTQ format sequences
2. **minimap2-omp** - DNA sequences
3. **leukocyte-omp** - Video file (may skip)

### Priority 4: CFD Data (2 benchmarks)
**Complex format** - may need documentation

1. **cfd-omp** - CFD domain format
2. **minibude-omp** - Molecular docking format

---

## Implementation Plan

### Phase 1: Quick Wins (Day 1)
Generate simple numerical/binary data for 12 benchmarks:

```bash
# hotspot3D-omp
mkdir -p ../data/hotspot3D
python3 generate_hotspot_data.py  # 512x8 matrices

# svd3x3-omp
python3 generate_svd_data.py  # 1M random 3x3 matrices

# permutate-omp
dd if=/dev/urandom of=data/truerand_1bit.bin bs=1M count=10

# Image benchmarks
python3 generate_test_images.py  # Generate PPM/BMP images
```

### Phase 2: Graph Data (Day 2)
Generate graph structures:

```python
# bfs-omp: Generate 1M vertex graph
import networkx as nx
G = nx.random_graphs.barabasi_albert_graph(1000000, 6)
# Export to required format

# b+tree-omp: Generate B-tree commands
# Generate INSERT/SEARCH/DELETE operations

# lanczos-omp: Social network graph
# Generate scale-free network

# kmeans-omp: Clustering data
# Generate random points in N dimensions
```

### Phase 3: Specialized Data (Day 3)
Handle domain-specific formats:

- **diamond-omp**: Generate synthetic FASTQ sequences
- **minimap2-omp**: Generate DNA strings
- **cfd-omp**: Research domain file format
- **minibude-omp**: Download from miniBUDE repo

---

## Data Generation Scripts

### Script 1: Hotspot3D Data
```python
#!/usr/bin/env python3
# generate_hotspot3d_data.py
import numpy as np
import os

os.makedirs('../data/hotspot3D', exist_ok=True)

# Generate 512x8 power matrix
power = np.random.rand(512, 8).astype(np.float32)
power.tofile('../data/hotspot3D/power_512x8')

# Generate 512x8 temperature matrix
temp = np.random.rand(512, 8).astype(np.float32) * 100 + 300  # 300-400K
temp.tofile('../data/hotspot3D/temp_512x8')

print("✅ Generated hotspot3D data files")
```

### Script 2: SVD3x3 Data
```python
#!/usr/bin/env python3
# generate_svd3x3_data.py
import numpy as np

# Generate 1M random 3x3 matrices
n_matrices = 1000000
matrices = np.random.randn(n_matrices, 3, 3).astype(np.float32)

with open('../svd3x3-cuda/Dataset_1M.txt', 'w') as f:
    for mat in matrices:
        for row in mat:
            f.write(' '.join(map(str, row)) + '\n')

print("✅ Generated SVD3x3 data (1M matrices)")
```

### Script 3: BFS Graph
```python
#!/usr/bin/env python3
# generate_bfs_graph.py
import os

os.makedirs('../data/bfs', exist_ok=True)

# Generate simple graph: 1M vertices, ~6M edges
n_vertices = 1000000
n_edges = 6000000

with open('../data/bfs/graph1MW_6.txt', 'w') as f:
    f.write(f"{n_vertices} {n_edges}\n")

    # Generate edges (Barabasi-Albert style)
    import random
    for i in range(n_edges):
        src = random.randint(0, n_vertices-1)
        dst = random.randint(0, n_vertices-1)
        weight = random.randint(1, 100)
        f.write(f"{src} {dst} {weight}\n")

print("✅ Generated BFS graph (1M vertices, 6M edges)")
```

### Script 4: Test Images
```python
#!/usr/bin/env python3
# generate_test_images.py
from PIL import Image
import numpy as np
import os

def generate_ppm(filename, width, height):
    """Generate a test PPM image"""
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    with open(filename, 'wb') as f:
        f.write(f'P6\n{width} {height}\n255\n'.encode())
        f.write(img.tobytes())

def generate_bmp(filename, width, height):
    """Generate a test BMP image"""
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(img).save(filename, 'BMP')

# boxfilter-omp
os.makedirs('boxfilter-omp/data', exist_ok=True)
generate_ppm('boxfilter-omp/data/lenaRGB.ppm', 512, 512)

# dxtc2-omp
os.makedirs('dxtc2-omp/data', exist_ok=True)
generate_ppm('dxtc2-omp/data/lena_std.ppm', 512, 512)
generate_ppm('dxtc2-omp/data/teapot512_std.ppm', 512, 512)

# medianfilter-omp
os.makedirs('medianfilter-omp/data', exist_ok=True)
generate_ppm('medianfilter-omp/data/SierrasRGB.ppm', 1024, 768)

# sobel-omp
os.makedirs('../sobel-sycl', exist_ok=True)
generate_bmp('../sobel-sycl/SobelFilter_Input.bmp', 512, 512)

print("✅ Generated test images")
```

---

## Next Steps

1. **Create data generation scripts** (above)
2. **Run scripts to generate data files**
3. **Test each benchmark** to verify it runs
4. **Document data requirements** in each benchmark's README
5. **Add scripts to repository** for reproducibility

## Expected Results

After fixing missing data files:
- **Current:** 250/326 working (77%)
- **After data fixes:** ~271/326 working (83%)
- **Impact:** +21 benchmarks (+6 percentage points)

Combined with the timeout fixes already validated, this brings the true success rate to **~83%** for OpenMP GPU offloading.
