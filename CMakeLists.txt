cmake_minimum_required(VERSION 3.25)
project(ia_project3)

set(CMAKE_CXX_STANDARD 17)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED OFF)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.81.0 COMPONENTS filesystem regex)
find_package(Armadillo REQUIRED)

if(BOOST_FOUND AND ARMADILLO_FOUND)
    include_directories(${BOOST_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})
    add_executable(ia_project3 main.cpp)
    target_link_libraries(ia_project3 ${BOOST_LIBRARIES} ${ARMADILLO_LIBRARIES})
endif()

