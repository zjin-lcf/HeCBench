NVCC = /opt/nvidia/hpc_sdk/Linux_aarch64/25.11/compilers/bin/nvc++
NVCC_FLAGS = -std=c++17 -O3 -mp=gpu

INC_DIR = include
SRC_DIR = src
OBJ_DIR = bin
EXEC_NAME = main

SRC_FILES := $(wildcard $(SRC_DIR)/*.cc)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cc,$(OBJ_DIR)/%.o,$(SRC_FILES))

$(OBJ_DIR)/$(EXEC_NAME): $(OBJ_FILES) 
	$(NVCC) $(NVCC_FLAGS) $(OBJ_FILES) -o $(OBJ_DIR)/$(EXEC_NAME)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	$(NVCC) $(NVCC_FLAGS) -I $(INC_DIR) -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(RM_FLAGS) $(OBJ_DIR)/*.o
	rm -f $(RM_FLAGS) $(OBJ_DIR)/*.obj
	rm -f $(RM_FLAGS) $(OBJ_DIR)/$(EXEC_NAME)

run: $(OBJ_DIR)/$(EXEC_NAME)
	$(LAUNCHER) ./$(OBJ_DIR)/$(EXEC_NAME) 10
