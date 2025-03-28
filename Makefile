NVCC = nvcc
CC = gcc

INCLUDE_DIR = include
SOURCE_DIR = src
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
EXE = $(BUILD_DIR)/analysis

# SOURCES := $(wildcard $(SOURCE_DIR)/*.cu) $(wildcard $(SOURCE_DIR)/*.c)
SOURCES := $(wildcard $(SOURCE_DIR)/*.cu)


OBJS := $(patsubst $(SOURCE_DIR/%.cu), $(OBJ_DIR)/%.o, $(SOURCES))
# OBJS += $(patsubst $(SOURCE_DIR/%.c), $(OBJ_DIR)/%.o, $(SOURCES))

CFLAGS = -I$(INCLUDE_DIR) -Xcompiler -Wall
LDFLAGS = 

all: $(EXE)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: $(SOURCE_DIR)/%.cu | $(OBJ_DIR) 
	$(NVCC) $(CFLAGS) -c $< -o $@

# $(OBJ_DIR)/%.o: $(SOURCE_DIR)/%.c | $(OBJ_DIR)
# 	$(CC) $(CFLAGS) -c $< -o $@

$(EXE): $(OBJS)
	$(NVCC) $(LDFLAGS) $^ -o $@


clean:
	rm -rf $(OBJ_DIR) $(EXE)
