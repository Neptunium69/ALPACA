cmake_minimum_required(VERSION 3.0.2)
project(alpaca_msgs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  nav_msgs
  sensor_msgs
  std_msgs
)

add_message_files(
  FILES
  ObjectDetection.msg
  ObservationFrame.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  geometry_msgs
  message_runtime
  nav_msgs
  sensor_msgs
  std_msgs
)
