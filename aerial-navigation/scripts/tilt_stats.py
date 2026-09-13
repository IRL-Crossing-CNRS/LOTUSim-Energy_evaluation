"""Cruise attitude statistics from a PX4 ULog.

Prints: n mean sd p50 p95 max %>44   (degrees), or NODATA below 50 samples.

Altitude is INTERPOLATED onto the attitude timestamps, not index-paired.
vehicle_attitude_groundtruth logs at ~57 Hz and vehicle_local_position_
groundtruth at 50 Hz, so pairing sample i of one with sample i of the other
drifts by (1 - 50/57.4) of elapsed time — about 25 s by the end of a 190 s
flight — and truncating to the shorter array silently discards the whole tail.
Both errors bias an out-and-back mission badly, because its two legs are flown
into and with the wind and so sit at very different tilts.
"""
import sys, numpy as np
from pyulog import ULog


def stats(path, alt_min=45.0):
    u = ULog(path, ['vehicle_attitude_groundtruth', 'vehicle_local_position_groundtruth'])
    a = u.get_dataset('vehicle_attitude_groundtruth').data
    ta = np.array(a['timestamp'], dtype=float) * 1e-6
    q = np.stack([a['q[0]'], a['q[1]'], a['q[2]'], a['q[3]']], axis=1); w, x, y, z = q.T
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    tilt = np.degrees(np.arccos(np.clip(np.cos(roll)*np.cos(pitch), -1, 1)))
    p = u.get_dataset('vehicle_local_position_groundtruth').data
    tp = np.array(p['timestamp'], dtype=float) * 1e-6
    alt = np.interp(ta, tp, -np.array(p['z']))
    cr = alt > alt_min
    if cr.sum() < 50:
        return None
    t = tilt[cr]
    return dict(n=int(cr.sum()), mean=t.mean(), sd=t.std(ddof=1), p50=np.percentile(t, 50),
                p95=np.percentile(t, 95), mx=t.max(), sat=100*(t > 44).mean())


if __name__ == '__main__':
    s = stats(sys.argv[1])
    print("NODATA" if s is None else
          f"{s['n']:6d} {s['mean']:6.2f} {s['sd']:6.2f} {s['p50']:6.2f} {s['p95']:6.2f} {s['mx']:6.2f} {s['sat']:6.1f}")
