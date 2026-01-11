#!/usr/bin/env python3

import sys
import select
import termios
import tty

import rospy
from geometry_msgs.msg import Twist

MOVE_BINDINGS = {
	"w": (1, 0),
	"s": (-1, 0),
	"a": (0, 1),
	"d": (0, -1),
	"q": (1, 1),
	"e": (1, -1),
	"z": (-1, 1),
	"c": (-1, -1),
}


def get_key(settings):
	tty.setraw(sys.stdin.fileno())
	rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
	key = ""
	if rlist:
		key = sys.stdin.read(1)
	termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
	return key


def main():
	rospy.init_node("keyboard_smoother", anonymous=True)
	pub = rospy.Publisher("/smoother_cmd_vel", Twist, queue_size=1)
	pub_2 = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
	speed = rospy.get_param("~speed", 0.1)
	turn = rospy.get_param("~turn", 1.0)
	rate = rospy.Rate(60)

	settings = termios.tcgetattr(sys.stdin)
	rospy.loginfo("Use WASD or QE/ZC to drive; Space to stop; Ctrl-C to exit.")

	try:
		while not rospy.is_shutdown():
			key = get_key(settings)
			twist = Twist()

			if key == "\x03":
				break

			if key in MOVE_BINDINGS:
				linear, angular = MOVE_BINDINGS[key]
				twist.linear.x = linear * speed
				twist.angular.z = angular * turn
			elif key == " ":
				twist = Twist()
			elif key:
				rospy.logdebug("ignored key %r", key)

			pub.publish(twist)
			pub_2.publish(twist)
			rate.sleep()
	except rospy.ROSInterruptException:
		pass
	finally:
		pub.publish(Twist())
		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


if __name__ == "__main__":
	main()
