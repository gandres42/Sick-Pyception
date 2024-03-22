from networktables import NetworkTables
import time
import numpy as np
import rigid_body_motion as rbm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import directed_hausdorff as hausdorff
import math

NetworkTables.initialize(server='localhost')
nt = NetworkTables.getTable('Perception')

mount_rotation = np.array(R.from_euler('xyz', (150, 0, 0), degrees=True).as_matrix(), dtype=np.float32)
max_dst = -math.inf

try:
    prev_cloud = None
    dt = np.array([0, 0, 0])
    dr = np.array([0, 0, 0, 0])
    sum_cloud = np.empty(shape=(0, 3), dtype=np.float32)

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
        cloud = np.delete(cloud, -1, axis=1)
        cloud = np.take(cloud, landmark_points, 0)
        cloud = cloud @ mount_rotation
            
        # only do the expensive bits when theres actually a new point cloud
        if prev_cloud is not None and not np.array_equal(cloud, prev_cloud):
            # hausdorff distance between unmodified cloud and map
            d, i1, i2 = hausdorff(cloud, sum_cloud)

            # update odometry for use in map extension initial estimate
            t, r = rbm.iterative_closest_point(prev_cloud, cloud)
            # rot = np.array(R.from_quat([r[1], r[2], r[3], r[0]]).as_matrix(), dtype=np.float32)
            trans = np.array(t, dtype=np.float32)
            dt = dt + trans
            dr = r
            
            # if map isn't empty, update cloud to match it and add to map if original distance meets threshold
            if len(sum_cloud) > 0:
                t, r = rbm.iterative_closest_point(sum_cloud, cloud, init_transform=(dt, dr))
                rot = np.array(R.from_quat([r[1], r[2], r[3], r[0]]).as_matrix(), dtype=np.float32)
                trans = np.array(t, dtype=np.float32)
                cloud = cloud @ rot
                cloud = cloud + trans
            if d > 1:
                print("appending " + str(time.monotonic()))
                sum_cloud = np.concatenate((sum_cloud, cloud), axis=0)

            # add color entry to cloud, prev_cloud and push to networktables
            color = np.full((len(cloud)), np.frombuffer(np.array([150, 0, 255, 100], dtype=np.uint8).tobytes(), dtype=np.float32))
            prev_color = np.full((len(prev_cloud)), np.frombuffer(np.array([0, 255, 0, 100], dtype=np.uint8).tobytes(), dtype=np.float32))
            sum_color = np.full((len(sum_cloud)), np.frombuffer(np.array([255, 255, 255, 100], dtype=np.uint8).tobytes(), dtype=np.float32))
            nt_cloud = np.c_[cloud, color]
            prev_nt_cloud = np.c_[prev_cloud, prev_color]
            sum_nt_cloud = np.c_[sum_cloud, sum_color]


            nt.putRaw('slam', np.concatenate((nt_cloud, sum_nt_cloud),axis=0).tobytes())
        prev_cloud = cloud
except KeyboardInterrupt:
    pass
