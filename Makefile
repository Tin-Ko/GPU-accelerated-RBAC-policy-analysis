NVCC = nvcc
CXX = g++

INCLUDE_DIR = include
SOURCE_DIR = src
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
EXE = $(BUILD_DIR)/main

CU_SOURCES := $(wildcard $(SOURCE_DIR)/*.cu)
CPP_SOURCES := $(wildcard $(SOURCE_DIR)/*.cpp)
# SOURCES := $(wildcard $(SOURCE_DIR)/*.cu)


CU_OBJS := $(patsubst $(SOURCE_DIR)/%.cu, $(OBJ_DIR)/%.cu.o, $(CU_SOURCES))
CPP_OBJS := $(patsubst $(SOURCE_DIR)/%.cpp, $(OBJ_DIR)/%.cpp.o, $(CPP_SOURCES))

OBJS := $(CU_OBJS) $(CPP_OBJS)

# Test closure
TEST_CLOSURE_OBJS := $(OBJ_DIR)/testClosure.cu.o $(OBJ_DIR)/closure.cu.o
TEST_CLOSURE_EXE := $(BUILD_DIR)/testClosure

NVCCFLAGS = -I$(INCLUDE_DIR) -Xcompiler -Wall
CFLAGS = -I$(INCLUDE_DIR) -Wall
LDFLAGS = 

# Build target
all: $(EXE)


# Create build directories if doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Compile CUDA source files
$(OBJ_DIR)/%.cu.o: $(SOURCE_DIR)/%.cu | $(OBJ_DIR) 
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C++ source files
$(OBJ_DIR)/%.cpp.o: $(SOURCE_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CFLAGS) -c $< -o $@

# Link all objects into final executable
$(EXE): $(OBJS)
	$(NVCC) $(LDFLAGS) $^ -o $@
# Clean build
clean:
	rm -rf $(BUILD_DIR)/*
