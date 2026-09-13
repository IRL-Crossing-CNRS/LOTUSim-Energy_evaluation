#!/usr/bin/env python3
"""Record live run assertions into LOG_DIR/run_assertions.json.

Must run WHILE the scenario is up: the wake-region count is only readable from
a latched topic that dies with the run, so it cannot be recovered afterwards.
Source ../env.sh first or the message types will not import.

    python3 probe_run.py <log_dir> [--seconds 10]
"""
import json, sys, time, argparse

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--world", default="energy")
    a = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from lotusim_msgs.msg import Wind as WindMsg, WindRegionArray, WindTurbineArray

    rclpy.init()
    n = Node("run_probe")
    latched = QoSProfile(depth=1)
    latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
    wind, regs, turb = [], [], []
    n.create_subscription(WindMsg, "/aerialWorld/wind", wind.append, 10)
    n.create_subscription(WindRegionArray, "/aerialWorld/wind/regions", regs.append, latched)
    n.create_subscription(WindTurbineArray, f"/{a.world}/wind/turbines", turb.append, latched)

    t0 = time.time()
    while time.time() - t0 < a.seconds:
        rclpy.spin_once(n, timeout_sec=0.2)

    out = {
        "wind_msgs": len(wind),
        "wind_vector": [wind[-1].linear_velocity.x, wind[-1].linear_velocity.y] if wind else None,
        "wind_live": len(wind) > 1,
        "wake_regions": len(regs[-1].regions) if regs else 0,
        # Peak turbulence amplitude actually on the wire. 0.0 means either the
        # scenario left `turbulence` off or the field never made it out of the
        # SDK — either way the drone is flying a deterministic wake.
        "region_sigma_max": max((r.turbulence_sigma for r in regs[-1].regions), default=0.0) if regs else 0.0,
        "turbines": len(turb[-1].turbines) if turb else 0,
        "farm_power_w": sum(t.power_w for t in turb[-1].turbines) if turb else 0.0,
    }
    path = f"{a.log_dir.rstrip('/')}/run_assertions.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(json.dumps(out))
    rclpy.shutdown()
    return 0

if __name__ == "__main__":
    sys.exit(main())
