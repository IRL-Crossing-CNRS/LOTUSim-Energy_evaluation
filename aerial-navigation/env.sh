#!/usr/bin/env bash
# Environment for the analysis and driver scripts in this study.
#
#   source aerial-navigation/env.sh
#
# `scenario_launch.sh` sets up its own environment, so launching a scenario
# needs no sourcing. Everything else here does: probe_run.py and
# wake_field_topview.py import `lotusim_msgs`, and the drivers read the world
# file out of the core repository.
#
# Two variables locate the workspaces, and both may be preset before sourcing:
#
#   LOTUSIM_WS            the LOTUSim core workspace   (default ~/lotusim_ws)
#   LOTUSIM_SCENARIO_WS   the LOTUSim-generic-scenario workspace
#
# Order matters. Colcon's prepend-unique dedupes an entry that is already in a
# path variable instead of moving it to the front, so sourcing the core
# workspace is not enough to make it win: whatever the shell already had stays
# ahead. A second LOTUSim install left on PYTHONPATH that way is the failure
# this guards against, and it is quiet — the Python module resolves from one
# workspace and the C typesupport from the other, which imports cleanly and
# then misbehaves somewhere unrelated. The explicit prepends below put this
# core workspace in front of every resolution path after all sourcing is done.

export LOTUSIM_WS="${LOTUSIM_WS:-$HOME/lotusim_ws}"
export LOTUSIM_PATH="${LOTUSIM_PATH:-$LOTUSIM_WS/src/LOTUSim}"
# Trailing slash required: xdyn concatenates it with a model YAML's relative
# "mesh:" path, and without it every relative mesh path fails to resolve.
export LOTUSIM_MODELS_PATH="$LOTUSIM_PATH/assets/models/"
export LOTUSIM_SCENARIO_WS="${LOTUSIM_SCENARIO_WS:-$HOME/Documents/workspace/lotusim/LOTUSim-generic-scenario}"

source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"

if [ -f "$LOTUSIM_PATH/install/setup.bash" ]; then
  source "$LOTUSIM_PATH/install/setup.bash"
elif [ -f "$LOTUSIM_WS/install/setup.bash" ]; then
  source "$LOTUSIM_WS/install/setup.bash"
else
  echo "no core build found under $LOTUSIM_PATH/install or $LOTUSIM_WS/install" >&2
fi
source "$LOTUSIM_SCENARIO_WS/install/setup.bash"

_prefix="$LOTUSIM_PATH/install"
[ -d "$_prefix" ] || _prefix="$LOTUSIM_WS/install"
export PYTHONPATH="$_prefix/lib/python$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$_prefix/lib:${LD_LIBRARY_PATH:-}"
export AMENT_PREFIX_PATH="$_prefix:${AMENT_PREFIX_PATH:-}"
unset _prefix

# A second LOTUSim overlay on the path is the quiet failure described above.
if [ "$(python3 - <<'PY' 2>/dev/null
import os
seen = {os.path.realpath(p) for p in os.environ.get("PYTHONPATH", "").split(":")
        if p and os.path.isdir(os.path.join(p, "lotusim_msgs"))}
print(len(seen))
PY
)" -gt 1 ]; then
  echo "warning: more than one lotusim_msgs is visible on PYTHONPATH." >&2
  echo "         Remove the other LOTUSim install from this shell before running." >&2
fi
