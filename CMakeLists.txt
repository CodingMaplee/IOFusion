cmake_minimum_required(VERSION 3.10)
project(MainLoop)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fpermissive -pthread -lrt -fPIC -O3")#-fsanitize=address -g -march=native
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(VISUALIZATION "Build With Pangolin" ON)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10.2")

find_package(CUDA REQUIRED)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS -arch=sm_62;-std=c++14;-lm)

ADD_DEFINITIONS(-fpermissive)

find_package (Threads)
find_package (Eigen3 REQUIRED)
find_package (OpenCV 4.5.5 REQUIRED)
MESSAGE(STATUS "OpenCV version: "${OpenCV_VERSION})

list( APPEND CMAKE_MODULE_PATH /home/AiLab/2022AI/g2o-master/cmake_modules)
find_package (G2O REQUIRED)
find_package (Sophus REQUIRED)
find_package (PythonLibs  REQUIRED)

message("Using python:  ${PYTHONLIBS_VERSION_STRING} ")
message("Using PYTHON_LIBRARIES:  ${PYTHON_LIBRARIES} ")
message("Using PYTHON_INCLUDE_DIRS:  ${PYTHON_INCLUDE_DIRS} ")

file(GLOB_RECURSE cpp_file src/*.cpp)
file(GLOB_RECURSE cu_file src/*.cu)
#file(GLOB_RECURSE cu_file src/DynamicReconstruction/cuda/*.cu)
if(VISUALIZATION)
  find_package(Pangolin QUIET)
  message("-- found PANGOLIN, add visualization plug.")
  if(Pangolin_FOUND)
    find_package(Boost COMPONENTS system thread)
    add_definitions(-DWITH_VISUALIZATION)
    include_directories(
      ${Pangolin_INCLUDE_DIRS}
      ${Boost_INCLUDE_DIRS}
      ${PROJECT_SOURCE_DIR}/include/Visualization/
    )
  endif()
else()
  message("-- not found PANGOLIN, visualization not enable.")
  file(GLOB_RECURSE Visualization_CPP_Files src/PangolinOutputWrapper.cpp)
  file(GLOB_RECURSE Visualization_CU_Files src/PangolinOutputWrapper.cu)
  list(REMOVE_ITEM cpp_file ${Visualization_CPP_Files})
  list(REMOVE_ITEM cu_file ${Visualization_CU_Files}) 
endif()

include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIR}
        ${SOPHUS_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/include/
        ${PROJECT_SOURCE_DIR}/include/mLib/include
        ${PROJECT_SOURCE_DIR}/include/mLibExternal
        ${PROJECT_SOURCE_DIR}/include/cutil/cutil/inc/
        ${PROJECT_SOURCE_DIR}/include/FreeImage/
        ${PROJECT_SOURCE_DIR}/include/interface/
        ${PROJECT_SOURCE_DIR}/include/DepthSensing/
        ${PROJECT_SOURCE_DIR}/include/Solver/
        ${PROJECT_SOURCE_DIR}/include/OrbSlam3/
        ${PROJECT_SOURCE_DIR}/include/OrbSlam3/CameraModels
        ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
        ${PYTHON_INCLUDE_DIRS}
)
cuda_add_library(MainLoop  SHARED
        ${cu_file}
        ${cpp_file}
        )

SET(G2O_LIBRARIES ${G2O_STUFF_LIBRARY} ${G2O_CORE_LIBRARY} ${G2O_CLI_LIBRARY} ${G2O_SOLVER_CHOLMOD} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_CSPARSE_EXTENSION}
        ${G2O_SOLVER_DENSE} ${G2O_SOLVER_PCG} ${G2O_SOLVER_SLAM2D_LINEAR} ${G2O_SOLVER_STRUCTURE_ONLY} ${G2O_SOLVER_EIGEN} ${G2O_TYPES_DATA} ${G2O_TYPES_ICP} ${G2O_TYPES_SBA}
        ${G2O_TYPES_SCLAM2D} ${G2O_TYPES_SIM3} ${G2O_TYPES_SLAM2D} ${G2O_TYPES_SLAM3D})


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/SLAMLib)
add_library(SLAMLib SHARED
        src/OrbSlam3/System.cc
        src/OrbSlam3/Tracking.cc
        src/OrbSlam3/LocalMapping.cc
        src/OrbSlam3/LoopClosing.cc
        src/OrbSlam3/ORBextractor.cc
        src/OrbSlam3/ORBmatcher.cc
        src/OrbSlam3/FrameDrawer.cc
        src/OrbSlam3/Converter.cc
        src/OrbSlam3/MapPoint.cc
        src/OrbSlam3/KeyFrame.cc
        src/OrbSlam3/Atlas.cc
        src/OrbSlam3/Map.cc
        src/OrbSlam3/MapDrawer.cc
        src/OrbSlam3/Optimizer.cc
        src/OrbSlam3/Frame.cc
        src/OrbSlam3/KeyFrameDatabase.cc
        src/OrbSlam3/Sim3Solver.cc
        src/OrbSlam3/Viewer.cc
        src/OrbSlam3/ImuTypes.cc
        src/OrbSlam3/G2oTypes.cc
        src/OrbSlam3/CameraModels/Pinhole.cpp
        src/OrbSlam3/CameraModels/KannalaBrandt8.cpp
        src/OrbSlam3/OptimizableTypes.cpp
        src/OrbSlam3/MLPnPsolver.cpp
        src/OrbSlam3/GeometricTools.cc
        src/OrbSlam3/TwoViewReconstruction.cc
        src/OrbSlam3/Config.cc
        src/OrbSlam3/Settings.cc
        include/OrbSlam3/System.h
        include/OrbSlam3/Tracking.h
        include/OrbSlam3/LocalMapping.h
        include/OrbSlam3/LoopClosing.h
        include/OrbSlam3/ORBextractor.h
        include/OrbSlam3/ORBmatcher.h
        include/OrbSlam3/FrameDrawer.h
        include/OrbSlam3/Converter.h
        include/OrbSlam3/MapPoint.h
        include/OrbSlam3/KeyFrame.h
        include/OrbSlam3/Atlas.h
        include/OrbSlam3/Map.h
        include/OrbSlam3/MapDrawer.h
        include/OrbSlam3/Optimizer.h
        include/OrbSlam3/Frame.h
        include/OrbSlam3/KeyFrameDatabase.h
        include/OrbSlam3/Sim3Solver.h
        include/OrbSlam3/Viewer.h
        include/OrbSlam3/ImuTypes.h
        include/OrbSlam3/G2oTypes.h
        include/OrbSlam3/CameraModels/GeometricCamera.h
        include/OrbSlam3/CameraModels/Pinhole.h
        include/OrbSlam3/CameraModels/KannalaBrandt8.h
        include/OrbSlam3/OptimizableTypes.h
        include/OrbSlam3/MLPnPsolver.h
        include/OrbSlam3/GeometricTools.h
        include/OrbSlam3/TwoViewReconstruction.h
        include/OrbSlam3/SerializationUtils.h
        include/OrbSlam3/Config.h
        include/OrbSlam3/Settings.h)

#add_subdirectory(Thirdparty/g2o)


target_link_libraries(MainLoop
  cuda 
  freeimage
  SLAMLib
  #${OPT_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBS}
  ${G2O_LIBRARIES}
  ${SOPHUS_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
  #${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
  -lboost_serialization
  -lcrypto
  #${MyOpenMeshCore} OpenMeshTools #--- openmesh libs
)

if(Pangolin_FOUND)
  target_link_libraries(MainLoop
    ${Pangolin_LIBRARIES}
    boost_system 
    boost_thread
  )
endif()



add_executable(dyna_fusion_example example/example.cpp)

target_link_libraries(dyna_fusion_example
        MainLoop
        ${PYTHON_LIBRARIES}
)

