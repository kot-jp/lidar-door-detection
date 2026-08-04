cmake_minimum_required(VERSION 3.8)
project(skid_robot_sim)

find_package(ament_cmake REQUIRED)

# URDFやlaunchファイルをインストール
install(
  DIRECTORY urdf worlds launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
