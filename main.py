from networktables import NetworkTables
import time
import numpy as np
import rigid_body_motion as rbm
from scipy.spatial.transform import Rotation as R


NetworkTables.initialize(server='localhost')
nt = NetworkTables.getTable('Perception')

base_rotation = R.from_euler('xyz', (150, 0, 0), degrees=True).as_matrix()

try:
    prev_cloud = None
    yaw = 0
    pitch = 0
    roll = 0
    x = 0
    y = 0
    z = 0

    while True:
        b = nt.getRaw('filtered points', None)
        if b is None:
            continue

        cloud = np.frombuffer(b, dtype=np.float32)
        cloud.shape = (int(len(cloud)/4), 4)
        colors = np.frombuffer(np.array(cloud[:,3]).tobytes(), dtype=np.uint8)
        colors.shape = (int(len(colors)/4), 4)

        landmark_points = []
        for i in range(len(colors)):
            if colors[i, 0] == 0: landmark_points.append(i)
        cloud = np.take(cloud, landmark_points, 0)
        cloud = np.delete(cloud, -1, axis=1)
        cloud_len = len(cloud)
        for i in range(cloud_len):
            cloud[i] = np.dot(cloud[i], base_rotation)
        
        if np.array_equal(cloud, prev_cloud):
            continue

        

        # if prev_cloud is not None:
        #     t, r = rbm.iterative_closest_point(prev_cloud, cloud)
        #     print()
        #     # print(r)
        #     angles = R.from_quat(r).as_euler('xyz', degrees=True)
        #     yaw = (yaw + angles[0]) % 360
        #     pitch = (pitch + angles[1]) % 360
        #     roll = (roll + angles[2]) % 360
        #     print(yaw, pitch, roll)
        #     # print(t)
        #     x = x + t[0]
        #     y = y + t[1]
        #     z = z + t[2]
        #     print(x, y, z)

        if prev_cloud is not None:
            color = np.full((len(cloud)), np.frombuffer(np.array([150, 0, 255, 100], dtype=np.uint8).tobytes(), dtype=np.float32))
            prev_color = np.full((len(prev_cloud)), np.frombuffer(np.array([0, 255, 0, 100], dtype=np.uint8).tobytes(), dtype=np.float32))
            nt_cloud = np.c_[cloud, color]
            prev_nt_cloud = np.c_[prev_cloud, prev_color]
            nt.putRaw('filter only points', np.concatenate((nt_cloud, prev_nt_cloud),axis=0).tobytes())
        prev_cloud = cloud
except KeyboardInterrupt:
    pass
