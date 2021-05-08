CC        = dpcpp
CC_FLAGS = -DUSE_GPU -std=c++14 -O3 -Wall -I../include -pthread

BIN = -o bin

INC_DIR = include
SRC_DIR = src
OBJ_DIR = bin
EXEC_NAME = demo

SRC_FILES := $(wildcard $(SRC_DIR)/*.cc)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cc,$(OBJ_DIR)/%.o,$(SRC_FILES))


$(OBJ_DIR)/$(EXEC_NAME): $(OBJ_FILES)
	$(CC) $(CC_FLAGS) $(OBJ_FILES) -o $(OBJ_DIR)/$(EXEC_NAME)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	$(CC) $(CC_FLAGS) -I $(INC_DIR) -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(RM_FLAGS) $(OBJ_DIR)/*.o
	rm -f $(RM_FLAGS) $(OBJ_DIR)/*.obj
	rm -f $(RM_FLAGS) $(OBJ_DIR)/$(EXEC_NAME)

run: $(OBJ_DIR)/$(EXEC_NAME)
	./$(OBJ_DIR)/$(EXEC_NAME) 10
