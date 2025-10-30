cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

project(evo-movement VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")
# Build a library from the source files (exclude main.cpp so tests can link to internals)
set(LIB_SRC ${SRC_FILES})
list(REMOVE_ITEM LIB_SRC ${PROJECT_SOURCE_DIR}/src/main.cpp)
add_library(evo_movement_lib STATIC ${LIB_SRC})
target_include_directories(evo_movement_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)

# build the executable from main and link the library
add_executable(evo-movement-sim ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(evo-movement-sim PRIVATE evo_movement_lib)

enable_testing()

# Tests
file(GLOB TEST_SRC "${PROJECT_SOURCE_DIR}/tests/*.cpp")
foreach(tsrc ${TEST_SRC})
	get_filename_component(tname ${tsrc} NAME_WE)
	add_executable(${tname} ${tsrc})
	target_include_directories(${tname} PRIVATE ${PROJECT_SOURCE_DIR}/include)
	target_link_libraries(${tname} PRIVATE evo_movement_lib)
	# Register test with full path to the built target so ctest finds it for
	# multi-configuration builds (e.g. Visual Studio / MSVC). Use generator
	# expression to get the target file at build time.
	add_test(NAME ${tname} COMMAND $<TARGET_FILE:${tname}>)

	# Add labels to tests so users can run subsets via ctest -L <label>
	# Default label: unit. Mark the config print test as a smoke test too.
	if(tname STREQUAL "test_config_print")
		set_tests_properties(${tname} PROPERTIES LABELS "unit;smoke")
	else()
		set_tests_properties(${tname} PROPERTIES LABELS "unit")
	endif()
endforeach()

# Optional: target to clean generated webviewer output (frames, snapshots, nn_index, etc.)
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
	add_custom_target(clean-webviewer
		COMMAND ${CMAKE_COMMAND} -E echo "Cleaning webviewer output using ${Python3_EXECUTABLE}..."
		COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/clean_webviewer.py --dry-run
		WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
		COMMENT "Show what webviewer files would be removed (use --yes to actually remove)"
	)
else()
	add_custom_target(clean-webviewer
		COMMAND ${CMAKE_COMMAND} -E echo "Python not found; cannot run webviewer cleaner."
		COMMENT "Install Python or run scripts/clean_webviewer.py manually"
	)
endif()
