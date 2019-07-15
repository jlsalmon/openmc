cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
project(openmc C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Setup output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set module path
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

#===============================================================================
# Command line options
#===============================================================================

option(openmp   "Enable shared-memory parallelism with OpenMP"   ON)
option(profile  "Compile with profiling flags"                   OFF)
option(debug    "Compile with debug flags"                       OFF)
option(optimize "Turn on all compiler optimization flags"        OFF)
option(coverage "Compile with coverage analysis flags"           OFF)
option(dagmc    "Enable support for DAGMC (CAD) geometry"        OFF)

#===============================================================================
# MPI for distributed-memory parallelism
#===============================================================================

set(MPI_ENABLED FALSE)
if($ENV{CXX} MATCHES "(mpi[^/]*|CC)$")
  message(STATUS "Detected MPI wrapper: $ENV{CXX}")
  set(MPI_ENABLED TRUE)
endif()

#===============================================================================
# HDF5 for binary output
#===============================================================================

# Allow user to specify HDF5_ROOT
if (NOT (CMAKE_VERSION VERSION_LESS 3.12))
  cmake_policy(SET CMP0074 NEW)
endif()

# Unfortunately FindHDF5.cmake will always prefer a serial HDF5 installation
# over a parallel installation if both appear on the user's PATH. To get around
# this, we check for the environment variable HDF5_ROOT and if it exists, use it
# to check whether its a parallel version.

if(NOT DEFINED HDF5_PREFER_PARALLEL)
  if(DEFINED ENV{HDF5_ROOT} AND EXISTS $ENV{HDF5_ROOT}/bin/h5pcc)
    set(HDF5_PREFER_PARALLEL TRUE)
  else()
    set(HDF5_PREFER_PARALLEL FALSE)
  endif()
endif()

find_package(HDF5 REQUIRED COMPONENTS C HL)
if(HDF5_IS_PARALLEL)
  if(NOT MPI_ENABLED)
    message(FATAL_ERROR "Parallel HDF5 must be used with MPI.")
  endif()
  message(STATUS "Using parallel HDF5")
endif()

#===============================================================================
# DAGMC Geometry Support - need DAGMC/MOAB
#===============================================================================
if(dagmc)
  find_package(DAGMC REQUIRED)
  link_directories(${DAGMC_LIBRARY_DIRS})
endif()

#===============================================================================
# pugixml library
#===============================================================================

add_library(pugixml vendor/pugixml/pugixml.cpp)
target_include_directories(pugixml PUBLIC vendor/pugixml/)

#===============================================================================
# xtensor header-only library
#===============================================================================

add_subdirectory(vendor/xtl)
add_subdirectory(vendor/xtensor)
target_link_libraries(xtensor INTERFACE xtl)


#===============================================================================
# CUDA
#===============================================================================

# For Xcode 5, gcc is actually clang, so we have to tell CUDA to treat the compiler as
# clang, so that it doesn't mistake it for something else.
if(USING_CLANG_C)
  set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
endif()

find_package(CUDA 7.5 REQUIRED)
find_package(OpenGL REQUIRED)

# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)


# Add some useful default arguments to the nvcc flags.  This is an example of how we use
# PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE and following
# block of code will not be executed leaving you free to edit the values as much as you
# wish from the GUI or from ccmake.
if(NOT PASSED_FIRST_CONFIGURE)

  list(APPEND default_nvcc_flags "--use_fast_math" "-arch sm_75" "-lineinfo" )

  if (CUDA_VERSION VERSION_LESS "3.0")
    list(APPEND default_nvcc_flags "--keep")
  endif()

  if( APPLE )
    # Undef'ing __BLOCKS__ for OSX builds.  This is due to a name clash between OSX 10.6
    # C headers and CUDA headers
    list(APPEND default_nvcc_flags "-U__BLOCKS__")
  endif()

  # Append to nvcc flags
  foreach(flag IN LISTS default_nvcc_flags)
    list(FIND CUDA_NVCC_FLAGS ${flag} index)
    if(index EQUAL -1)
      list(APPEND CUDA_NVCC_FLAGS ${flag})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    endif()
  endforeach(flag)

endif(NOT PASSED_FIRST_CONFIGURE)

# Add required nvcc flag for callable programs under CUDA 8
if (CUDA_VERSION VERSION_GREATER "7.5")
  set(flag "--keep-device-functions")
  list(FIND CUDA_NVCC_FLAGS ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS ${flag})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
  endif()
endif()

# Create a flag for mac which will allow apps to add the local cuda toolkit
# install path to the app's rpath.
if( APPLE )
  set( CUDA_TOOLKIT_RPATH_FLAG "-Wl,-rpath,${CUDA_TOOLKIT_ROOT_DIR}/lib" )
endif()


find_package(OptiX REQUIRED)
include_directories(
        "${OptiX_INCLUDE}"
        include
        ${HDF5_INCLUDE_DIRS}
        vendor/pugixml/
)

##################################################################
# SUtil compilation

set(SAMPLES_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx")
set(SAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(CUDA_GENERATED_OUTPUT_DIR ${SAMPLES_PTX_DIR})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/sampleConfig.h.in ${CMAKE_CURRENT_BINARY_DIR}/sampleConfig.h @ONLY)

# Path to sutil.h that all the samples need
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/sutil
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OptiX_INCLUDE}/optixu
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CUDA_INCLUDE_DIRS} )


include(FindSUtilGLUT)

set(SAMPLES_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/device_include)
set(SAMPLES_CUDA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuda)

##################################################################
# IMGUI compilation
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/support )
#add_subdirectory( support/imgui )

##################################################################
# GLFW compilation
#set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
#set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
#set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
#add_subdirectory( support/glfw )

#########################################################
# OPTIX_add_sample_executable
#
# Convience function for adding samples to the code.  You can copy the contents of this
# funtion into your individual project if you wish to customize the behavior.  Note that
# in CMake, functions have their own scope, whereas macros use the scope of the caller.
function(OPTIX_add_sample_executable target_name)

  # These calls will group PTX and CUDA files into their own directories in the Visual
  # Studio projects.
#  source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
#  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")

  # Separate the sources from the CMake and CUDA options fed to the macro.  This code
  # comes from the CUDA_COMPILE_PTX macro found in FindCUDA.cmake.  We are copying the
  # code here, so that we can use our own name for the target.  target_name is used in the
  # creation of the output file names, and we want this to be unique for each target in
  # the SDK.
  CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

  # Create the rules to build the PTX from the CUDA files.
  CUDA_WRAP_SRCS( ${target_name} PTX generated_files ${source_files} ${cmake_options}
          OPTIONS ${options} )

  # Here is where we create the rule to make the executable.  We define a target name and
  # list all the source files used to create the target.  In addition we also pass along
  # the cmake_options parsed out of the arguments.
#  add_executable(${target_name}
#          ${source_files}
#          ${generated_files}
#          ${cmake_options}
#          )

  # Most of the samples link against sutil, optix, and glfw.  Here is the
  # rule that specifies this linkage.
#  target_link_libraries( ${target_name}
#          sutil_sdk
#          optix
#          glfw
#          imgui
#          ${OPENGL_gl_LIBRARY}
#          ${optix_rpath}
#          )
#  if(USING_GNU_CXX)
#    target_link_libraries( ${target_name} m ) # Explicitly link against math library (C samples don't do that by default)
#  endif()
endfunction()

add_subdirectory(sutil)

# Compile the cuda files to ptx.  Note that this will ignore all of the non CUDA
# files.
set(sources
        src/optix/basic.cu
        src/optix/simulation.cu
        src/optix/material.cu
        src/optix/geometry.cu
        src/optix/particle.cu
        src/optix/physics.cu
        src/optix/variables.cu
        src/optix/cell.cu
        src/optix/math_functions.cu
        src/optix/source.cu
        src/optix/nuclide.cu
        src/optix/random_lcg.cu
        src/optix/endf.cu
        src/optix/distribution.cu
        src/optix/distribution_angle.cu
        src/optix/distribution_energy.cu
        src/optix/secondary_uncorrelated.cu
        src/optix/secondary_kalbach.cu
        src/optix/reaction_product.cu
        src/optix/render/accum_camera.cu
        src/optix/render/constantbg.cu
        src/optix/render/normal_shader.cu
        src/optix/render/optixGeometryTriangles.cu)
if(NOT CUDA_NVRTC_ENABLED)
  CUDA_COMPILE_PTX(ptx_files ${sources})
endif()


#target_link_libraries( ${target_name}
##        sutil_sdk
#        optix
##        glfw
##        imgui
##        ${OPENGL_gl_LIBRARY}
##        ${optix_rpath}
#        )

# Now that everything is done, indicate that we have finished configuring at least once.
# We use this variable to set certain defaults only on the first pass, so that we don't
# continually set them over and over again.
set(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "Already Configured once?")


#===============================================================================
# Set compile/link flags based on which compiler is being used
#===============================================================================

if(openmp)
  # Requires CMake 3.1+
  find_package(OpenMP)
  if(OPENMP_FOUND)
    list(APPEND cxxflags ${OpenMP_CXX_FLAGS})
    list(APPEND ldflags ${OpenMP_CXX_FLAGS})
  endif()
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND cxxflags -O2)
if(debug)
  list(REMOVE_ITEM cxxflags -O2)
  list(APPEND cxxflags -g -O0)
endif()
if(profile)
  list(APPEND cxxflags -g -fno-omit-frame-pointer)
endif()
if(optimize)
  list(REMOVE_ITEM cxxflags -O2)
  list(APPEND cxxflags -O3)
endif()

# Show flags being used
message(STATUS "OpenMC C++ flags: ${cxxflags}")
message(STATUS "OpenMC Linker flags: ${ldflags}")


#===============================================================================
# RPATH information
#===============================================================================

# This block of code ensures that dynamic libraries can be found via the RPATH
# whether the executable is the original one from the build directory or the
# installed one in CMAKE_INSTALL_PREFIX. Ref:
# https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling

# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

#===============================================================================
# faddeeva library
#===============================================================================

add_library(faddeeva STATIC vendor/faddeeva/Faddeeva.cc)
target_include_directories(faddeeva PUBLIC vendor/faddeeva/)
target_compile_options(faddeeva PRIVATE ${cxxflags})

#===============================================================================
# libopenmc
#===============================================================================

add_library(libopenmc SHARED
  src/bank.cpp
  src/bremsstrahlung.cpp
  src/dagmc.cpp
  src/cell.cpp
  src/cmfd_solver.cpp
  src/cross_sections.cpp
  src/distribution.cpp
  src/distribution_angle.cpp
  src/distribution_energy.cpp
  src/distribution_multi.cpp
  src/distribution_spatial.cpp
  src/eigenvalue.cpp
  src/endf.cpp
  src/error.cpp
  src/initialize.cpp
  src/finalize.cpp
  src/geometry.cpp
  src/geometry_aux.cpp
  src/hdf5_interface.cpp
  src/lattice.cpp
  src/material.cpp
  src/math_functions.cpp
  src/mesh.cpp
  src/message_passing.cpp
  src/mgxs.cpp
  src/mgxs_interface.cpp
  src/nuclide.cpp
  src/output.cpp
  src/particle.cpp
  src/particle_restart.cpp
  src/photon.cpp
  src/physics.cpp
  src/physics_common.cpp
  src/physics_mg.cpp
  src/plot.cpp
  src/position.cpp
  src/progress_bar.cpp
  src/random_lcg.cpp
  src/reaction.cpp
  src/reaction_product.cpp
  src/scattdata.cpp
  src/secondary_correlated.cpp
  src/secondary_kalbach.cpp
  src/secondary_nbody.cpp
  src/secondary_uncorrelated.cpp
  src/settings.cpp
  src/simulation.cpp
  src/source.cpp
  src/state_point.cpp
  src/string_utils.cpp
  src/summary.cpp
  src/surface.cpp
  src/tallies/derivative.cpp
  src/tallies/filter.cpp
  src/tallies/filter_azimuthal.cpp
  src/tallies/filter_cellborn.cpp
  src/tallies/filter_cellfrom.cpp
  src/tallies/filter_cell.cpp
  src/tallies/filter_delayedgroup.cpp
  src/tallies/filter_distribcell.cpp
  src/tallies/filter_energyfunc.cpp
  src/tallies/filter_energy.cpp
  src/tallies/filter_legendre.cpp
  src/tallies/filter_material.cpp
  src/tallies/filter_mesh.cpp
  src/tallies/filter_meshsurface.cpp
  src/tallies/filter_mu.cpp
  src/tallies/filter_particle.cpp
  src/tallies/filter_polar.cpp
  src/tallies/filter_sph_harm.cpp
  src/tallies/filter_sptl_legendre.cpp
  src/tallies/filter_surface.cpp
  src/tallies/filter_universe.cpp
  src/tallies/filter_zernike.cpp
  src/tallies/tally.cpp
  src/tallies/tally_scoring.cpp
  src/tallies/trigger.cpp
  src/optix/optix_cell.cpp
  src/optix/optix_surface.cpp
  src/optix/optix_geometry.cpp
  src/optix/optix_renderer.cpp
  src/optix/optix_data.cpp
  src/timer.cpp
  src/thermal.cpp
  src/track_output.cpp
  src/urr.cpp
  src/volume_calc.cpp
  src/wmp.cpp
  src/xml_interface.cpp
  src/xsdata.cpp ${ptx_files})

set_target_properties(libopenmc PROPERTIES
  OUTPUT_NAME openmc)

target_include_directories(libopenmc
  PUBLIC include ${HDF5_INCLUDE_DIRS})

# Set compile flags
target_compile_options(libopenmc PRIVATE ${cxxflags})

if (HDF5_IS_PARALLEL)
  target_compile_definitions(libopenmc PRIVATE -DPHDF5)
endif()
if (MPI_ENABLED)
  target_compile_definitions(libopenmc PUBLIC -DOPENMC_MPI)
endif()

# Set git SHA1 hash as a compile definition
execute_process(COMMAND git rev-parse HEAD
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SHA1_SUCCESS
                OUTPUT_VARIABLE GIT_SHA1
                ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
if(GIT_SHA1_SUCCESS EQUAL 0)
  target_compile_definitions(libopenmc PRIVATE -DGIT_SHA1="${GIT_SHA1}")
endif()

# target_link_libraries treats any arguments starting with - but not -l as
# linker flags. Thus, we can pass both linker flags and libraries together.
target_link_libraries(libopenmc ${ldflags} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES}
                      pugixml faddeeva xtensor optix sutil_sdk)

if(dagmc)
  target_compile_definitions(libopenmc PRIVATE DAGMC)
  target_link_libraries(libopenmc ${DAGMC_LIBRARIES})
  target_include_directories(libopenmc PRIVATE ${DAGMC_INCLUDE_DIRS})
endif()

#===============================================================================
# openmc executable
#===============================================================================
add_executable(openmc src/main.cpp)
target_compile_options(openmc PRIVATE ${cxxflags})
target_link_libraries(openmc libopenmc)

# Ensure C++14 standard is used. Starting with CMake 3.8, another way this could
# be done is using the cxx_std_14 compiler feature.
set_target_properties(
    openmc libopenmc faddeeva pugixml
    PROPERTIES CXX_STANDARD 14 CXX_EXTENSIONS OFF)

#===============================================================================
# Python package
#===============================================================================

add_custom_command(TARGET libopenmc POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  $<TARGET_FILE:libopenmc>
  ${CMAKE_CURRENT_SOURCE_DIR}/openmc/capi/$<TARGET_FILE_NAME:libopenmc>
  COMMENT "Copying libopenmc to Python module directory")

#===============================================================================
# Install executable, scripts, manpage, license
#===============================================================================

install(TARGETS openmc libopenmc
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )
install(DIRECTORY src/relaxng DESTINATION share/openmc)
install(FILES man/man1/openmc.1 DESTINATION share/man/man1)
install(FILES LICENSE DESTINATION "share/doc/openmc" RENAME copyright)
install(DIRECTORY include/ DESTINATION include)
