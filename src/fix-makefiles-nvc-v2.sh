#!/bin/bash

# Script to update all -omp Makefiles to use nvc++ compiler (v2)
# Handles icpc, icpx, and flag compatibility issues
# Usage: ./fix-makefiles-nvc-v2.sh

NVC_PATH="/opt/nvidia/hpc_sdk/Linux_aarch64/25.11/compilers/bin/nvc++"
COUNT=0

echo "Updating all -omp Makefiles for nvc++ compatibility..."
echo ""

# Find all -omp directories with Makefiles
for DIR in $(find . -maxdepth 1 -type d -name "*-omp" | sort); do
    MAKEFILE="$DIR/Makefile"

    if [ -f "$MAKEFILE" ]; then
        UPDATED=0

        # Backup if not already backed up
        if [ ! -f "$MAKEFILE.bak" ]; then
            cp "$MAKEFILE" "$MAKEFILE.bak"
        fi

        # Replace icpx with nvc++
        if grep -q "^CC.*=.*icpx" "$MAKEFILE"; then
            sed -i "s|^CC\s*=\s*icpx|CC        = $NVC_PATH|g" "$MAKEFILE"
            UPDATED=1
        fi

        # Replace icpc with nvc++
        if grep -q "^CC.*=.*icpc" "$MAKEFILE"; then
            sed -i "s|^CC\s*=\s*icpc|CC        = $NVC_PATH|g" "$MAKEFILE"
            UPDATED=1
        fi

        # Replace Intel OpenMP flags with NVIDIA flags
        if grep -q -- "-fiopenmp -fopenmp-targets=spir64" "$MAKEFILE"; then
            sed -i 's|-fiopenmp -fopenmp-targets=spir64|-mp=gpu|g' "$MAKEFILE"
            UPDATED=1
        fi

        # Replace Intel OpenMP flags (alternative format)
        if grep -q -- "-qopenmp -fopenmp-targets=spir64" "$MAKEFILE"; then
            sed -i 's|-qopenmp -fopenmp-targets=spir64|-mp=gpu|g' "$MAKEFILE"
            UPDATED=1
        fi

        # Replace standalone -fopenmp with -mp=multicore for CPU
        if grep -q -- "-fopenmp" "$MAKEFILE" && ! grep -q -- "-mp=" "$MAKEFILE"; then
            sed -i 's|-fopenmp|-mp=multicore|g' "$MAKEFILE"
            UPDATED=1
        fi

        # Replace -qopenmp with -mp=multicore
        if grep -q -- "-qopenmp" "$MAKEFILE"; then
            sed -i 's|-qopenmp|-mp=multicore|g' "$MAKEFILE"
            UPDATED=1
        fi

        # Remove -D__STRICT_ANSI__
        if grep -q -- "-D__STRICT_ANSI__" "$MAKEFILE"; then
            sed -i 's| -D__STRICT_ANSI__||g' "$MAKEFILE"
            UPDATED=1
        fi

        # Replace -ffast-math with -fast (nvc++ equivalent)
        if grep -q -- "-ffast-math" "$MAKEFILE"; then
            sed -i 's|-ffast-math|-fast|g' "$MAKEFILE"
            UPDATED=1
        fi

        # Replace -qnextgen (Intel flag) - just remove it
        if grep -q -- "-qnextgen" "$MAKEFILE"; then
            sed -i 's| -qnextgen||g' "$MAKEFILE"
            UPDATED=1
        fi

        if [ $UPDATED -eq 1 ]; then
            echo "Updated $MAKEFILE"
            ((COUNT++))
        fi
    fi
done

echo ""
echo "Updated $COUNT Makefiles"
echo "Backups saved as Makefile.bak"
echo ""
echo "To restore originals: find . -name 'Makefile.bak' -exec bash -c 'mv \"\$1\" \"\${1%.bak}\"' _ {} \;"
