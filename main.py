from networktables import NetworkTables
import time
import numpy as np
import matplotlib.pyplot as plt

NetworkTables.initialize(server='localhost')
nt = NetworkTables.getTable('Perception')

try:
    while True:
        b = nt.getRaw('filtered points', None)
        if b is None:
            continue

        a = np.frombuffer(b, dtype=np.float32)
        a.shape = (int(len(a)/4), 4)
        c = np.frombuffer(np.array(a[:,3]).tobytes(), dtype=np.uint8)
        c.shape = (int(len(c)/4), 4)

        landmark_points = []
        for i in range(len(c)):
            if c[i, 0] == 0:
                landmark_points.append(i)
        pc = np.take(a, landmark_points, 0)
        nt.putRaw('filter only points', pc.tobytes())
        
        print(len(pc), time.monotonic())
except KeyboardInterrupt():
    pass
