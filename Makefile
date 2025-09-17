# Simple Makefile for Transformer LLM
# Alternative to CMake for systems that prefer Make

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
DEBUGFLAGS = -g -O0 -DDEBUG
RELEASEFLAGS = -O3 -DNDEBUG

# Metal GPU acceleration flags (macOS only)
ifeq ($(shell uname -s),Darwin)
    METAL_FLAGS = -framework Metal -framework MetalKit -framework MetalPerformanceShaders -framework Foundation
    GPU_SOURCES = metal_gpu_simple.mm
    CXXFLAGS += -DMETAL_ENABLED
else
    METAL_FLAGS = 
    GPU_SOURCES = 
endif

# Source files
CORE_SOURCES = matrix_ops.cpp attention.cpp transformer_blocks.cpp tokenizer_embedding.cpp transformer_model.cpp $(GPU_SOURCES)
DEMO_SOURCES = main.cpp $(CORE_SOURCES)
TRAINING_SOURCES = train.cpp training.cpp $(CORE_SOURCES)
INFERENCE_SOURCES = inference.cpp training.cpp $(CORE_SOURCES)

DEMO_OBJECTS = $(DEMO_SOURCES:.cpp=.o)
TRAINING_OBJECTS = $(TRAINING_SOURCES:.cpp=.o)
INFERENCE_OBJECTS = $(INFERENCE_SOURCES:.cpp=.o)

TARGETS = transformer_demo train_model inference train_demo benchmark
PRIMARY_TARGET = transformer_demo

# Default target
all: $(TARGETS)

# Debug build
debug: CXXFLAGS += $(DEBUGFLAGS)
debug: $(TARGETS)

# Release build  
release: CXXFLAGS += $(RELEASEFLAGS)
release: $(TARGETS)

# Link the executables
transformer_demo: $(DEMO_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(METAL_FLAGS)

train_model: $(TRAINING_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(METAL_FLAGS)

inference: $(INFERENCE_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(METAL_FLAGS)

train_demo: train_demo.o training.o $(CORE_SOURCES:.cpp=.o)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(METAL_FLAGS)

benchmark: benchmark_fixed.o training.o $(CORE_SOURCES:.cpp=.o)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(METAL_FLAGS)

# Compile source files
%.o: %.cpp transformer.h training.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile Objective-C++ files
%.o: %.mm transformer.h training.h metal_gpu_simple.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile Metal shaders (optional, for precompilation)
matrix_ops.metallib: matrix_ops.metal
	xcrun -sdk macosx metal -c matrix_ops.metal -o matrix_ops.air
	xcrun -sdk macosx metallib matrix_ops.air -o matrix_ops.metallib

# Clean build files
clean:
	rm -f *.o $(TARGETS) *.air *.metallib

# GPU-specific targets
gpu: CXXFLAGS += -DFORCE_GPU
gpu: $(TARGETS)

# Compile Metal shaders
shaders: matrix_ops.metallib

# Run the demo
run: transformer_demo
	./transformer_demo

# Run training
train: train_model
	./train_model

# Run inference (with GPU support)
infer: inference
	./inference

# Run GPU-accelerated inference
infer-gpu: inference
	./inference --auto-gpu

# Run interactive inference with GPU
infer-interactive: inference
	./inference --interactive --auto-gpu

# Install (simple version)
install: $(TARGETS)
	mkdir -p ~/bin
	cp $(TARGETS) ~/bin/

.PHONY: all debug release clean run train infer infer-gpu infer-interactive install info

# Build information
info:
	@echo "Compiler: $(CXX)"
	@echo "Flags: $(CXXFLAGS)"
	@echo "Core Sources: $(CORE_SOURCES)"
	@echo "Targets: $(TARGETS)"
