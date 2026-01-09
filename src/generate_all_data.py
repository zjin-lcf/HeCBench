#!/usr/bin/env python3
"""
Generate all missing data files for HeCBench benchmarks
Run from the src/ directory
"""

import os
import numpy as np
from PIL import Image
import struct

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    print(f"  📁 Created directory: {path}")

def generate_hotspot3d_data():
    """Generate hotspot3D power and temperature files"""
    print("\n=== hotspot3D-omp ===")
    ensure_dir('../data/hotspot3D')

    # Generate 512x8 power matrix (binary float32)
    power = np.random.rand(512, 8).astype(np.float32) * 100  # 0-100W
    power.tofile('../data/hotspot3D/power_512x8')
    print(f"  ✅ Generated power_512x8 ({power.nbytes} bytes)")

    # Generate 512x8 temperature matrix (binary float32)
    temp = np.random.rand(512, 8).astype(np.float32) * 100 + 300  # 300-400K
    temp.tofile('../data/hotspot3D/temp_512x8')
    print(f"  ✅ Generated temp_512x8 ({temp.nbytes} bytes)")

def generate_svd3x3_data():
    """Generate SVD 3x3 matrix dataset"""
    print("\n=== svd3x3-omp ===")
    ensure_dir('../svd3x3-cuda')

    # Generate 1M random 3x3 matrices
    n_matrices = 1000000
    print(f"  Generating {n_matrices} random 3x3 matrices...")

    with open('../svd3x3-cuda/Dataset_1M.txt', 'w') as f:
        for i in range(n_matrices):
            mat = np.random.randn(3, 3).astype(np.float32)
            for row in mat:
                f.write(' '.join(map(str, row)) + ' ')
            f.write('\n')

            if (i + 1) % 100000 == 0:
                print(f"    {i + 1:,} matrices written...")

    print(f"  ✅ Generated Dataset_1M.txt")

def generate_permutate_data():
    """Generate random binary data for permutate benchmark"""
    print("\n=== permutate-omp ===")
    ensure_dir('permutate-omp/data')

    # Generate 10MB of random data
    size_mb = 10
    data = np.random.randint(0, 2, size_mb * 1024 * 1024, dtype=np.uint8)
    data.tofile('permutate-omp/data/truerand_1bit.bin')
    print(f"  ✅ Generated truerand_1bit.bin ({size_mb}MB)")

def generate_ppm_image(filename, width, height):
    """Generate a test PPM image"""
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    with open(filename, 'wb') as f:
        f.write(f'P6\n{width} {height}\n255\n'.encode())
        f.write(img.tobytes())

def generate_bmp_image(filename, width, height):
    """Generate a test BMP image"""
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(img, 'RGB').save(filename, 'BMP')

def generate_image_data():
    """Generate all test images"""
    print("\n=== Image Benchmarks ===")

    # boxfilter-omp
    ensure_dir('boxfilter-omp/data')
    generate_ppm_image('boxfilter-omp/data/lenaRGB.ppm', 512, 512)
    print("  ✅ boxfilter-omp: lenaRGB.ppm")

    # dxtc2-omp
    ensure_dir('dxtc2-omp/data')
    generate_ppm_image('dxtc2-omp/data/lena_std.ppm', 512, 512)
    generate_ppm_image('dxtc2-omp/data/teapot512_std.ppm', 512, 512)
    # Note: .dds files are DirectDraw Surface format - skip for now
    print("  ✅ dxtc2-omp: lena_std.ppm, teapot512_std.ppm")
    print("  ⚠️  dxtc2-omp: Skipping .dds files (complex format)")

    # medianfilter-omp
    ensure_dir('medianfilter-omp/data')
    generate_ppm_image('medianfilter-omp/data/SierrasRGB.ppm', 1024, 768)
    print("  ✅ medianfilter-omp: SierrasRGB.ppm")

    # sobel-omp
    ensure_dir('../sobel-sycl')
    generate_bmp_image('../sobel-sycl/SobelFilter_Input.bmp', 512, 512)
    print("  ✅ sobel-omp: SobelFilter_Input.bmp")

def generate_bfs_graph():
    """Generate BFS graph data"""
    print("\n=== bfs-omp ===")
    ensure_dir('../data/bfs')

    # Generate graph: 1M vertices, ~6M edges
    n_vertices = 1000000
    n_edges = 6000000

    print(f"  Generating graph ({n_vertices:,} vertices, {n_edges:,} edges)...")

    with open('../data/bfs/graph1MW_6.txt', 'w') as f:
        f.write(f"{n_vertices} {n_edges}\n")

        # Generate random edges (simple random graph)
        np.random.seed(42)
        sources = np.random.randint(0, n_vertices, n_edges)
        dests = np.random.randint(0, n_vertices, n_edges)
        weights = np.random.randint(1, 101, n_edges)

        for i, (src, dst, wt) in enumerate(zip(sources, dests, weights)):
            f.write(f"{src} {dst} {wt}\n")

            if (i + 1) % 1000000 == 0:
                print(f"    {i + 1:,} edges written...")

    print(f"  ✅ Generated graph1MW_6.txt")

def generate_btree_data():
    """Generate B-tree test data"""
    print("\n=== b+tree-omp ===")
    ensure_dir('../data/b+tree')

    # Generate 1M integers for insertion
    n_records = 1000000
    data = np.random.randint(0, 10000000, n_records)

    with open('../data/b+tree/mil.txt', 'w') as f:
        for val in data:
            f.write(f"{val}\n")
    print(f"  ✅ Generated mil.txt ({n_records:,} records)")

    # Generate command file (insert, search operations)
    with open('../data/b+tree/command.txt', 'w') as f:
        # Insert commands
        f.write("i\n")  # Insert mode
        for val in data[:1000]:  # First 1000 inserts
            f.write(f"{val}\n")

        # Search commands
        f.write("s\n")  # Search mode
        for val in data[:100]:  # Search for first 100
            f.write(f"{val}\n")
    print(f"  ✅ Generated command.txt")

def generate_kmeans_data():
    """Generate k-means clustering data"""
    print("\n=== kmeans-omp ===")
    ensure_dir('../data/kmeans')

    # Generate synthetic clustering data (34 dimensions, 494020 points)
    n_points = 494020
    n_dims = 34
    n_clusters = 5

    print(f"  Generating clustering data ({n_points:,} points, {n_dims} dims)...")

    # Generate clusters
    data = []
    for cluster in range(n_clusters):
        center = np.random.randn(n_dims) * 10
        points = np.random.randn(n_points // n_clusters, n_dims) + center
        data.append(points)

    data = np.vstack(data).astype(np.float32)

    with open('../data/kmeans/kdd_cup', 'w') as f:
        for point in data:
            f.write(' '.join(map(str, point)) + '\n')

    print(f"  ✅ Generated kdd_cup ({data.nbytes:,} bytes)")

def generate_lanczos_data():
    """Generate social network graph for Lanczos"""
    print("\n=== lanczos-omp ===")
    ensure_dir('lanczos-omp/data')

    # Generate social network (800K edges in edge list format)
    n_vertices = 100000
    n_edges = 800000

    print(f"  Generating social network ({n_vertices:,} vertices, {n_edges:,} edges)...")

    with open('lanczos-omp/data/social-large-800k.txt', 'w') as f:
        # Write header
        f.write(f"{n_vertices} {n_edges}\n")

        # Generate edges (power-law distribution for social network)
        np.random.seed(42)
        # Use preferential attachment-like distribution
        sources = np.random.power(2.5, n_edges) * (n_vertices - 1)
        dests = np.random.power(2.5, n_edges) * (n_vertices - 1)

        for src, dst in zip(sources.astype(int), dests.astype(int)):
            if src != dst:  # No self-loops
                f.write(f"{src} {dst}\n")

    print(f"  ✅ Generated social-large-800k.txt")

def generate_face_data():
    """Generate face detection classifier data"""
    print("\n=== face-omp ===")
    ensure_dir('../face-cuda')

    # Generate classifier info (simplified Viola-Jones cascade)
    with open('../face-cuda/info.txt', 'w') as f:
        # Write a simple cascade structure
        n_stages = 20
        f.write(f"{n_stages}\n")
        for stage in range(n_stages):
            n_classifiers = np.random.randint(10, 50)
            threshold = np.random.randn()
            f.write(f"{n_classifiers} {threshold}\n")
    print(f"  ✅ Generated info.txt")

    # Generate class labels
    with open('../face-cuda/class.txt', 'w') as f:
        # Write classifier features
        for i in range(1000):
            # Random Haar-like features
            x = np.random.randint(0, 24)
            y = np.random.randint(0, 24)
            w = max(1, np.random.randint(1, max(2, 24-x)))
            h = max(1, np.random.randint(1, max(2, 24-y)))
            threshold = np.random.randn()
            f.write(f"{x} {y} {w} {h} {threshold}\n")
    print(f"  ✅ Generated class.txt")

def generate_ss_data():
    """Generate string search input"""
    print("\n=== ss-omp ===")
    ensure_dir('../ss-sycl')

    # Generate text with searchable patterns
    with open('../ss-sycl/StringSearch_Input.txt', 'w') as f:
        # Generate 1MB of text
        words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog',
                 'pattern', 'search', 'algorithm', 'performance', 'benchmark']
        text = []
        for _ in range(200000):
            text.append(np.random.choice(words))
        f.write(' '.join(text))
    print(f"  ✅ Generated StringSearch_Input.txt")

def generate_ccs_data():
    """Generate bicluster data for CCS"""
    print("\n=== ccs-omp ===")
    ensure_dir('../ccs-cuda')

    # Generate gene expression matrix (100 genes x 100 samples)
    matrix = np.random.randn(100, 100).astype(np.float32)

    with open('../ccs-cuda/Data_Constant_100_1_bicluster.txt', 'w') as f:
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')
    print(f"  ✅ Generated Data_Constant_100_1_bicluster.txt")

def main():
    print("="*60)
    print("HeCBench Data Generation Script")
    print("="*60)
    print("\nGenerating missing data files for benchmarks...")

    # Priority 1: Easy synthetic data
    try:
        generate_hotspot3d_data()
    except Exception as e:
        print(f"  ❌ Error: {e}")

    try:
        generate_permutate_data()
    except Exception as e:
        print(f"  ❌ Error: {e}")

    try:
        generate_image_data()
    except Exception as e:
        print(f"  ❌ Error: {e}")

    try:
        generate_ss_data()
    except Exception as e:
        print(f"  ❌ Error: {e}")

    try:
        generate_face_data()
    except Exception as e:
        print(f"  ❌ Error: {e}")

    try:
        generate_ccs_data()
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Priority 2: Graph data (larger files, takes longer)
    try:
        generate_bfs_graph()
    except Exception as e:
        print(f"  ❌ Error: {e}")

    try:
        generate_btree_data()
    except Exception as e:
        print(f"  ❌ Error: {e}")

    try:
        generate_lanczos_data()
    except Exception as e:
        print(f"  ❌ Error: {e}")

    try:
        generate_kmeans_data()
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Note: SVD takes a while (1M matrices)
    # try:
    #     generate_svd3x3_data()
    # except Exception as e:
    #     print(f"  ❌ Error: {e}")

    print("\n" + "="*60)
    print("✅ Data generation complete!")
    print("="*60)
    print("\nGenerated data for:")
    print("  - hotspot3D-omp")
    print("  - permutate-omp")
    print("  - boxfilter-omp")
    print("  - medianfilter-omp")
    print("  - sobel-omp")
    print("  - dxtc2-omp (partial)")
    print("  - face-omp")
    print("  - ss-omp")
    print("  - ccs-omp")
    print("  - bfs-omp")
    print("  - b+tree-omp")
    print("  - lanczos-omp")
    print("  - kmeans-omp")
    print("\nNot generated (need more work):")
    print("  - svd3x3-omp (run separately, takes time)")
    print("  - cfd-omp (complex CFD format)")
    print("  - diamond-omp (FASTQ format)")
    print("  - leukocyte-omp (video file)")
    print("  - minimap2-omp (DNA sequences)")
    print("  - minibude-omp (molecular docking)")
    print("  - cmp-omp (seismic data)")
    print("  - hogbom-omp (radio astronomy)")

if __name__ == '__main__':
    main()
