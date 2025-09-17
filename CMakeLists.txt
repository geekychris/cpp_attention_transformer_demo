cmake_minimum_required(VERSION 3.14)
project(TransformerLLM VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# Debug mode flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")

# Release mode flags  
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# Source files
set(SOURCES
    main.cpp
    matrix_ops.cpp
    attention.cpp
    transformer_blocks.cpp
    tokenizer_embedding.cpp
    transformer_model.cpp
)

# Header files
set(HEADERS
    transformer.h
)

# Create the executable
add_executable(transformer_demo ${SOURCES} ${HEADERS})

# Set include directories
target_include_directories(transformer_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Optional: Add OpenMP for potential parallelization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(transformer_demo PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found and linked")
else()
    message(STATUS "OpenMP not found - compiling without parallelization")
endif()

# Create a simple Makefile alternative for systems without CMake
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/simple_build.sh.in"
    "${CMAKE_CURRENT_BINARY_DIR}/simple_build.sh"
    @ONLY
)

# Print build information
message(STATUS "Building TransformerLLM ${PROJECT_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# Installation rules
install(TARGETS transformer_demo
    RUNTIME DESTINATION bin
)

install(FILES ${HEADERS}
    DESTINATION include
)