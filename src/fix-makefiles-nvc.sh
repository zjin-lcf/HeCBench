#!/bin/bash

# Script to update all -omp Makefiles to use nvc++ compiler
# Usage: ./fix-makefiles-nvc.sh

NVC_PATH="/opt/nvidia/hpc_sdk/Linux_aarch64/25.11/compilers/bin/nvc++"
COUNT=0

echo "Updating all -omp Makefiles to use nvc++..."
echo ""

# Find all -omp directories with Makefiles
for DIR in $(find . -maxdepth 1 -type d -name "*-omp" | sort); do
    MAKEFILE="$DIR/Makefile"

    if [ -f "$MAKEFILE" ]; then
        # Check if it uses icpx
        if grep -q "^CC.*=.*icpx" "$MAKEFILE"; then
            echo "Updating $MAKEFILE..."

            # Backup original
            cp "$MAKEFILE" "$MAKEFILE.bak"

            # Replace icpx with nvc++ path
            sed -i "s|^CC\s*=\s*icpx|CC        = $NVC_PATH|g" "$MAKEFILE"

            # Replace Intel OpenMP flags with NVIDIA flags
            # -fiopenmp -fopenmp-targets=spir64 -> -mp=gpu
            sed -i 's|-fiopenmp -fopenmp-targets=spir64|-mp=gpu|g' "$MAKEFILE"

            # Replace standalone -fopenmp with -mp=multicore for CPU
            sed -i 's|-fopenmp|-mp=multicore|g' "$MAKEFILE"

            # Remove -D__STRICT_ANSI__ as it may not be needed for nvc++
            sed -i 's| -D__STRICT_ANSI__||g' "$MAKEFILE"

            ((COUNT++))
        fi
    fi
done

echo ""
echo "Updated $COUNT Makefiles"
echo "Backups saved as Makefile.bak"
echo ""
echo "To restore originals: find . -name 'Makefile.bak' -exec bash -c 'mv \"\$1\" \"\${1%.bak}\"' _ {} \;"
