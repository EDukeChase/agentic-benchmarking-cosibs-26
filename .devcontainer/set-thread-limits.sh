#!/usr/bin/env bash
# Cap the numeric thread pools at ~85% of the cores this container can see, so a
# benchmark run leaves the host usable. Computed at container start rather than
# hardcoded, so the same repo behaves sensibly on a 20-core desktop (17) and a
# 4-core laptop (3) with no per-machine setup.
#
# These variables are read by OpenMP, MKL, OpenBLAS, NumExpr and, through them,
# by NumPy, scikit-learn and PyTorch. Child processes started by the agent's
# execute_python tool inherit them, so generated benchmark code is covered too.
set -euo pipefail

CORES="$(nproc)"
LIMIT=$(( CORES * 85 / 100 ))
[ "$LIMIT" -lt 1 ] && LIMIT=1

PROFILE=/etc/profile.d/thread-limits.sh
cat > "$PROFILE" <<EOF
export OMP_NUM_THREADS=$LIMIT
export MKL_NUM_THREADS=$LIMIT
export OPENBLAS_NUM_THREADS=$LIMIT
export NUMEXPR_NUM_THREADS=$LIMIT
export VECLIB_MAXIMUM_THREADS=$LIMIT
EOF
chmod 0644 "$PROFILE"

# /etc/profile.d is only sourced by login shells; add it to bashrc so plain
# interactive terminals (what VS Code opens) pick it up as well.
if ! grep -q "thread-limits.sh" /root/.bashrc 2>/dev/null; then
    echo ". $PROFILE" >> /root/.bashrc
fi

echo "Thread limits set to $LIMIT of $CORES cores (85%)."
