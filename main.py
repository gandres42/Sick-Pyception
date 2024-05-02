import rospy
from nav_msgs.msg import Odometry
import rospy2 as rp2


def callback(data):
    print('-----------------------------')
    t = data.pose.pose.position
    r = data.pose.pose.orientation
    print(t)
    print()
    print(r)

def listener():
    rospy.init_node('localizer', anonymous=True)
    rospy.Subscriber('robot/dlio/odom_node/odom', Odometry, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()