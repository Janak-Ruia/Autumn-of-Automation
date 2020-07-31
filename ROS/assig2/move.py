#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
import math
rospy.init_node('my_initial', anonymous=True)

def move(velx, vely, velz, dist, forward):
    # Starts a new node
    

    velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
    vel_msg = Twist()

    #Receiveing the user's input
    print("Let's move your robot")

    #Checking if the movement is forward or backwards
    '''if(isForward):
                    vel_msg.linear.x = abs(speed)
                else:
                    vel_msg.linear.x = -abs(speed)
    '''#Since we are moving just in x-axis
    
    vel_msg.linear.x = velx
    vel_msg.linear.y = vely
    vel_msg.linear.z = velz
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    speed = math.sqrt(velx**2+vely**2+velz**2)
    #while not rospy.is_shutdown():

    #Setting the current time for distance calculus
    t0 = rospy.Time.now().to_sec()
    current_distance = 0
    #Loop to move the turtle in an specified distance
    while(current_distance < dist):
        #Publish the velocity
        

        velocity_publisher.publish(vel_msg)
        #Takes actual time to velocity calculus
        t1=rospy.Time.now().to_sec()
        #Calculates distancePoseStamped
        current_distance= speed*(t1-t0)

    #After the loop, stops the robot
    vel_msg.linear.x = 0
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    #Force the robot to stop
    velocity_publisher.publish(vel_msg)

def rotate(speed, angle, clockwise):

    velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
    vel_msg = Twist()

    print('lets turn your robot')
    angular_speed = speed*2*math.pi/360
    relative_angle = angle*2*math.pi/360
    #We wont use linear components
    vel_msg.linear.x=0
    vel_msg.linear.y=0
    vel_msg.linear.z=0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0

    if clockwise:
        vel_msg.angular.z = -abs(angular_speed)
    else:
        vel_msg.angular.z = abs(angular_speed)
    t0 = rospy.Time.now().to_sec()
    current_angle = 0

    while current_angle < relative_angle:
        velocity_publisher.publish(vel_msg)
        t1 = rospy.Time.now().to_sec()
        current_angle = angular_speed*(t1-t0)

    #Forcing our robot to stop
    vel_msg.angular.z = 0
    velocity_publisher.publish(vel_msg)
    #rospy.spin()
        


def make_j():
    move(1, 0, 0, 4, 1)
    rotate(180, 180, 1)
    move(1, 0, 0, 2, 1)
    rotate(90, 90, 0)
    #rotate(90, 90, 1)
    move(1, 0, 0, 4, 1)
    rotate(90, 90, 1)
    move(1, 0 ,0 , 1, 1)
    rotate(90, 90, 1)
    move(1, 0, 0, 1, 1)



if __name__ == '__main__':
    try:
        #Testing our function
        make_j()
    except rospy.ROSInterruptException: pass