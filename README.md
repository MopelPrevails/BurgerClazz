What Logan has cooked up

find_object.py - Lab1 Script, no longer being used
color_finder.py - Script that subscribes to the raw image topic and performs the hsv image transformation and color and threshold value selection. 
                  It outputs a float32 array where the first 3 floats are h,s,v and the last 3 are the corresponding thresholds
src/aizen_object_follower - Package Folder
src/aizen_object_follower/src/find_object.py - Node that runs on the robot. Subscribes to the raw image and the float32 containing the hsv values. It runs the thresholding and other processing techniques
                    used in the lab 1 code. It then publishes the same processed image we got in Lab 1 as a compressed image. It also publishes the centroid as a geometry_msgs/Point where z is 0.
processed_image_viewer.py - This node subscribes to the processed image and its code is essentially just the provided basic image viewer code given to us.

The build, install, and log folders are all for colcon build things. If running the package is giving you trouble maybe try removing these three folders (NOT src) and rerunning colcon build it fixed it for me

To run code on the robot,
  1. Navigate to the BurgerClazz Directory
  2. Run colcon build
  3. Run source install/local_setup.bash
  4. You can now run whatever package node you want. You can see what nodes you can run in object_follower with: ros2 pkg executables aizen_object_follower
  5. You can run a node with: ros2 run aizen_object_follower find_object.py  (Replace find_object.py with whateva ya node is)
  6. If you wanna check to see if shit is being published just open a new terminal and do: ros2 topic echo /topic (Replace /topic with whatever the topic name is i.e. /tracking/centroid)


TODO

make rotate_robot (script that subscribes to the centroid topic (/tracking/centroid) and publishes a twist to /cmd_vel)
make a launch file

Maybe add back the past center processing stuff? I took it out because I was having trouble setting it up with the formatting needed to write it as a class but could probably figure it out.
This is a low priority tho.
