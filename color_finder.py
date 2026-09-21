#!/usr/bin/env python3

# Imports
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

# Set size of the square around the click to sample the average color from
square_size = 11
half_square = square_size // 2

# Function to sample the HSV color from a square area around the clicked point
# Returns the median color and the color spread (standard deviation) in the sampled area
def sample_hsv(frame, x, y, square_size):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h, w = frame.shape[:2]

        y1 = max(0, y - half_square)
        y2 = min(h, y + half_square + 1)

        x1 = max(0, x - half_square)
        x2 = min(w, x + half_square + 1)

        square = hsv_frame[y1:y2, x1:x2]

        median_color = np.median(square, axis=(0, 1))
        color_spread = np.std(square.astype(np.float32), axis=(0, 1))

        return median_color, color_spread

# ColorFinder Node Class
class ColorFinder(Node):
	# Initialize the ColorFinder Node
	def __init__(self):		
		# Creates the node.
		super().__init__('color_finder')

		# Set Parameters
		self.declare_parameter('show_image_bool', True)
		self.declare_parameter('window_name', "Raw Image")
		self.declare_parameter('image_topic', "/image_raw/compressed")
		self.declare_parameter('hsv_topic', "/tracking/target_hsv")

		#Determine Window Showing Based on Input
		self._display_image = bool(self.get_parameter('show_image_bool').value)

		# Declare some variables
		self._titleOriginal = self.get_parameter('window_name').value # Image Window Title
		image_topic = self.get_parameter('image_topic').value # Image Topic
		hsv_topic = self.get_parameter('hsv_topic').value # HSV Topic	

		# Initialize variables
		self._imgBGR = None
		self._user_input = -1
		self.picked_color = None
		self.previous_center = None
		self.H_tolerance = None
		self.S_tolerance = None
		self.V_tolerance = None
		
		# Only create image frames if we are not running headless (_display_image sets this)
		if(self._display_image):
		# Set Up Image Viewing
			cv2.namedWindow(self._titleOriginal, cv2.WINDOW_AUTOSIZE) # Viewing Window
			cv2.moveWindow(self._titleOriginal, 50, 50) # Viewing Window Original Location

			cv2.setMouseCallback(self._titleOriginal, self.mouse_callback) # Set Mouse Callback for Picking Color
		
		#Set up QoS Profiles for passing images over WiFi
		image_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

		# Declare that the minimal_video_subscriber node is subcribing to the /camera/image/compressed topic.
		self._video_subscriber = self.create_subscription(
				CompressedImage,
				image_topic,
				self._image_callback,
				image_qos_profile)
		self._video_subscriber # Prevents unused variable warning.

		# Declare that the minimal_video_subscriber node is publishing to the /tracking/target_hsv topic.
		self._hsv_publisher = self.create_publisher(Float32MultiArray, hsv_topic, 10)
		self.get_logger().info(f"ColorFinder Node Initialized. Subscribed to {image_topic} and publishing to {hsv_topic}")

	# Callback function for the image subscriber, converts the compressed image to BGR format and stores it in _imgBGR
	def _image_callback(self, CompressedImage):	
		# The "CompressedImage" is transformed to a color image in BGR space and is store in "_imgBGR"
		# Convert compressed image data to numpy array
		np_arr = np.frombuffer(CompressedImage.data, np.uint8)
		# Decode image using OpenCV
		self._imgBGR = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

		# Don't Display the image if _imgBGR is None (if the image could not be decoded)
		if self._imgBGR is None:
			return

		# Display the image in a window if _display_image is True
		if(self._display_image):
			# Display the image in a window
			self.show_image(self._imgBGR)

	# Mouse callback function to pick color on click, sets the picked_color and tolerances based on the color spread in the sampled area
	def mouse_callback(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN and self._imgBGR is not None:

			# Get the current frame for color sampling
			frame = self._imgBGR

			# Sample the HSV color from the square area around the clicked point
			self.picked_color, color_spread = sample_hsv(
				frame,
				x,
				y,
				square_size
			)

			# Set tolerances based on the color spread, with limits to avoid too narrow or too wide tolerances
			self.H_tolerance = int(np.clip(
				8 + 2 * color_spread[0],
				8,
				25
			))

			self.S_tolerance = int(np.clip(
				30 + 2 * color_spread[1],
				50,
				100
			))

			self.V_tolerance = int(np.clip(
				30 + 2 * color_spread[2],
				50,
				100
			))

			self.get_logger().info(f"Picked HSV Color: {self.picked_color}, Tolerances: H={self.H_tolerance}, S={self.S_tolerance}, V={self.V_tolerance}")

			# Publish the picked color and tolerances to the HSV topic
			self.publish_hsv()

	# Function to publish the picked color and tolerances to the HSV topic
	def publish_hsv(self):
		if self.picked_color is not None:
			hsv_msg = Float32MultiArray()
			hsv_msg.data = [
				float(self.picked_color[0]),
				float(self.picked_color[1]),
				float(self.picked_color[2]),
				float(self.H_tolerance),
				float(self.S_tolerance),
				float(self.V_tolerance)
			]
			self._hsv_publisher.publish(hsv_msg)
			self.get_logger().info(f"Published HSV Color: {self.picked_color} with tolerances H={self.H_tolerance}, S={self.S_tolerance}, V={self.V_tolerance}")
		
	# Function to display the image in a window and wait for a key press
	def show_image(self, img):
		cv2.imshow(self._titleOriginal, img)
		# Cause a slight delay so image is displayed
		self._user_input=cv2.waitKey(50) #Use OpenCV keystroke grabber for delay.

	# Function to get the last user input key pressed
	def get_user_input(self):
		return self._user_input


# Main function to initialize the ROS2 node and run the ColorFinder
def main():
	rclpy.init() #init routine needed for ROS2.
	video_subscriber = ColorFinder() #Create class object to be used.

	# Spin the node to process callbacks and display images until 'q' is pressed
	while rclpy.ok():
		rclpy.spin_once(video_subscriber) # Trigger callback processing.
		if(video_subscriber._display_image):	
			if video_subscriber.get_user_input() == ord('q'):
				cv2.destroyAllWindows()
				break
            
	rclpy.logging.get_logger("Camera Viewer Node Info...").info("Shutting Down")
	#Clean up and shutdown.
	video_subscriber.destroy_node()  
	rclpy.shutdown()

if __name__ == '__main__':
	main()