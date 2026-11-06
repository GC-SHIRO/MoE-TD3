cmake_minimum_required(VERSION 3.8)
project(landing_sim)

find_package(ament_cmake REQUIRED)
# Gazebo Harmonic 对应的库版本。
find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)

add_library(DeckMotion SHARED src/DeckMotion.cc)
target_compile_features(DeckMotion PRIVATE cxx_std_17)
target_link_libraries(DeckMotion
  gz-sim8::gz-sim8
  gz-plugin2::register
  gz-transport13::gz-transport13
  gz-msgs10::gz-msgs10
)

install(TARGETS DeckMotion LIBRARY DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch models scripts DESTINATION share/${PROJECT_NAME})

ament_package()
