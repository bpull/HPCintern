# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/m175148/intern/Arcade-Learning-Environment-0.5.1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/m175148/intern/Arcade-Learning-Environment-0.5.1

# Include any dependencies generated for this target.
include CMakeFiles/sharedLibraryInterfaceExample.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sharedLibraryInterfaceExample.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sharedLibraryInterfaceExample.dir/flags.make

CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.o: CMakeFiles/sharedLibraryInterfaceExample.dir/flags.make
CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.o: doc/examples/sharedLibraryInterfaceExample.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/m175148/intern/Arcade-Learning-Environment-0.5.1/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.o -c /home/m175148/intern/Arcade-Learning-Environment-0.5.1/doc/examples/sharedLibraryInterfaceExample.cpp

CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/m175148/intern/Arcade-Learning-Environment-0.5.1/doc/examples/sharedLibraryInterfaceExample.cpp > CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.i

CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/m175148/intern/Arcade-Learning-Environment-0.5.1/doc/examples/sharedLibraryInterfaceExample.cpp -o CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.s

CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.o.requires:
.PHONY : CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.o.requires

CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.o.provides: CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.o.requires
	$(MAKE) -f CMakeFiles/sharedLibraryInterfaceExample.dir/build.make CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.o.provides.build
.PHONY : CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.o.provides

CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.o.provides.build: CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.o

# Object files for target sharedLibraryInterfaceExample
sharedLibraryInterfaceExample_OBJECTS = \
"CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.o"

# External object files for target sharedLibraryInterfaceExample
sharedLibraryInterfaceExample_EXTERNAL_OBJECTS =

doc/examples/sharedLibraryInterfaceExample: CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.o
doc/examples/sharedLibraryInterfaceExample: CMakeFiles/sharedLibraryInterfaceExample.dir/build.make
doc/examples/sharedLibraryInterfaceExample: /usr/lib/x86_64-linux-gnu/libSDLmain.a
doc/examples/sharedLibraryInterfaceExample: /usr/lib/x86_64-linux-gnu/libSDL.so
doc/examples/sharedLibraryInterfaceExample: CMakeFiles/sharedLibraryInterfaceExample.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable doc/examples/sharedLibraryInterfaceExample"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sharedLibraryInterfaceExample.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sharedLibraryInterfaceExample.dir/build: doc/examples/sharedLibraryInterfaceExample
.PHONY : CMakeFiles/sharedLibraryInterfaceExample.dir/build

CMakeFiles/sharedLibraryInterfaceExample.dir/requires: CMakeFiles/sharedLibraryInterfaceExample.dir/doc/examples/sharedLibraryInterfaceExample.cpp.o.requires
.PHONY : CMakeFiles/sharedLibraryInterfaceExample.dir/requires

CMakeFiles/sharedLibraryInterfaceExample.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sharedLibraryInterfaceExample.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sharedLibraryInterfaceExample.dir/clean

CMakeFiles/sharedLibraryInterfaceExample.dir/depend:
	cd /home/m175148/intern/Arcade-Learning-Environment-0.5.1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/m175148/intern/Arcade-Learning-Environment-0.5.1 /home/m175148/intern/Arcade-Learning-Environment-0.5.1 /home/m175148/intern/Arcade-Learning-Environment-0.5.1 /home/m175148/intern/Arcade-Learning-Environment-0.5.1 /home/m175148/intern/Arcade-Learning-Environment-0.5.1/CMakeFiles/sharedLibraryInterfaceExample.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sharedLibraryInterfaceExample.dir/depend

