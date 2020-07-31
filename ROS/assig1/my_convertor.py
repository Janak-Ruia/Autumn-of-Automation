#! /usr/bin/env python
import rospy
from assig_pqt.msg import quaternion, euler
import math

roll = pitch = yaw = 0.0

def convert(q):
    global roll, pitch, yaw
	#roll (x-axis rotation)
    sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    #pitch (y-axis rotation)
    sinp = 2 * (q.w * q.y - q.z * q.x)
    if math.abs(sinp) >= 1:
        pitch = math.copysign(math.pi, sinp)# use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    #yaw (z-axis rotation)
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    print(roll, pitch, yaw)


rospy.init_node('my_converter')
sub = rospy.Subscriber('topic1', quaternion, convert)
r = rospy.Rate(1)
pub = rospy.Publisher('topic2', euler, queue_size=10)

while not rospy.is_shutdown():
	euler_angles = []
	euler_angles.append(roll)
	euler_angles.append(pitch)
	euler_angles.append(yaw)
	rospy.loginfo(euler_angles)
	pub.publish(roll, pitch, yaw)
	r.sleep()
