# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/a/W/W.priv/Extern/cmake-3.13.4-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /home/a/W/W.priv/Extern/cmake-3.13.4-Linux-x86_64/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/cmake-build

# Include any dependencies generated for this target.
include CMakeFiles/OpenMPI_NonBlocking.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/OpenMPI_NonBlocking.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/OpenMPI_NonBlocking.dir/flags.make

CMakeFiles/OpenMPI_NonBlocking.dir/main.mpi.cpp.o: CMakeFiles/OpenMPI_NonBlocking.dir/flags.make
CMakeFiles/OpenMPI_NonBlocking.dir/main.mpi.cpp.o: ../main.mpi.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/cmake-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/OpenMPI_NonBlocking.dir/main.mpi.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/OpenMPI_NonBlocking.dir/main.mpi.cpp.o -c /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/main.mpi.cpp

CMakeFiles/OpenMPI_NonBlocking.dir/main.mpi.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OpenMPI_NonBlocking.dir/main.mpi.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/main.mpi.cpp > CMakeFiles/OpenMPI_NonBlocking.dir/main.mpi.cpp.i

CMakeFiles/OpenMPI_NonBlocking.dir/main.mpi.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OpenMPI_NonBlocking.dir/main.mpi.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/main.mpi.cpp -o CMakeFiles/OpenMPI_NonBlocking.dir/main.mpi.cpp.s

CMakeFiles/OpenMPI_NonBlocking.dir/MpiMaster.cpp.o: CMakeFiles/OpenMPI_NonBlocking.dir/flags.make
CMakeFiles/OpenMPI_NonBlocking.dir/MpiMaster.cpp.o: ../MpiMaster.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/cmake-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/OpenMPI_NonBlocking.dir/MpiMaster.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/OpenMPI_NonBlocking.dir/MpiMaster.cpp.o -c /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/MpiMaster.cpp

CMakeFiles/OpenMPI_NonBlocking.dir/MpiMaster.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OpenMPI_NonBlocking.dir/MpiMaster.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/MpiMaster.cpp > CMakeFiles/OpenMPI_NonBlocking.dir/MpiMaster.cpp.i

CMakeFiles/OpenMPI_NonBlocking.dir/MpiMaster.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OpenMPI_NonBlocking.dir/MpiMaster.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/MpiMaster.cpp -o CMakeFiles/OpenMPI_NonBlocking.dir/MpiMaster.cpp.s

CMakeFiles/OpenMPI_NonBlocking.dir/MpiSlave.cpp.o: CMakeFiles/OpenMPI_NonBlocking.dir/flags.make
CMakeFiles/OpenMPI_NonBlocking.dir/MpiSlave.cpp.o: ../MpiSlave.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/cmake-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/OpenMPI_NonBlocking.dir/MpiSlave.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/OpenMPI_NonBlocking.dir/MpiSlave.cpp.o -c /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/MpiSlave.cpp

CMakeFiles/OpenMPI_NonBlocking.dir/MpiSlave.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OpenMPI_NonBlocking.dir/MpiSlave.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/MpiSlave.cpp > CMakeFiles/OpenMPI_NonBlocking.dir/MpiSlave.cpp.i

CMakeFiles/OpenMPI_NonBlocking.dir/MpiSlave.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OpenMPI_NonBlocking.dir/MpiSlave.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/MpiSlave.cpp -o CMakeFiles/OpenMPI_NonBlocking.dir/MpiSlave.cpp.s

# Object files for target OpenMPI_NonBlocking
OpenMPI_NonBlocking_OBJECTS = \
"CMakeFiles/OpenMPI_NonBlocking.dir/main.mpi.cpp.o" \
"CMakeFiles/OpenMPI_NonBlocking.dir/MpiMaster.cpp.o" \
"CMakeFiles/OpenMPI_NonBlocking.dir/MpiSlave.cpp.o"

# External object files for target OpenMPI_NonBlocking
OpenMPI_NonBlocking_EXTERNAL_OBJECTS =

OpenMPI_NonBlocking: CMakeFiles/OpenMPI_NonBlocking.dir/main.mpi.cpp.o
OpenMPI_NonBlocking: CMakeFiles/OpenMPI_NonBlocking.dir/MpiMaster.cpp.o
OpenMPI_NonBlocking: CMakeFiles/OpenMPI_NonBlocking.dir/MpiSlave.cpp.o
OpenMPI_NonBlocking: CMakeFiles/OpenMPI_NonBlocking.dir/build.make
OpenMPI_NonBlocking: CMakeFiles/OpenMPI_NonBlocking.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/cmake-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable OpenMPI_NonBlocking"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/OpenMPI_NonBlocking.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/OpenMPI_NonBlocking.dir/build: OpenMPI_NonBlocking

.PHONY : CMakeFiles/OpenMPI_NonBlocking.dir/build

CMakeFiles/OpenMPI_NonBlocking.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/OpenMPI_NonBlocking.dir/cmake_clean.cmake
.PHONY : CMakeFiles/OpenMPI_NonBlocking.dir/clean

CMakeFiles/OpenMPI_NonBlocking.dir/depend:
	cd /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/cmake-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/cmake-build /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/cmake-build /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/cmake-build/CMakeFiles/OpenMPI_NonBlocking.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/OpenMPI_NonBlocking.dir/depend

