include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

BIN = -o bin

INC_DIR = include
SRC_DIR = src
OBJ_DIR = bin
EXEC_NAME = demo

SRC_FILES := $(wildcard $(SRC_DIR)/*.cc)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cc,$(OBJ_DIR)/%.o,$(SRC_FILES))

#===============================================================================
# Targets to Build
#===============================================================================

$(OBJ_DIR)/$(EXEC_NAME): $(OBJ_FILES)
	$(CC) $(CFLAGS) $(OBJ_FILES) -o $(OBJ_DIR)/$(EXEC_NAME)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	$(CC) $(CFLAGS) -I $(INC_DIR) -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(RM_FLAGS) $(OBJ_DIR)/*.o
	rm -f $(RM_FLAGS) $(OBJ_DIR)/*.obj
	rm -f $(RM_FLAGS) $(OBJ_DIR)/$(EXEC_NAME)

run: $(OBJ_DIR)/$(EXEC_NAME)
	./$(OBJ_DIR)/$(EXEC_NAME) 10
