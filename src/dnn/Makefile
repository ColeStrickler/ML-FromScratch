


CC=g++
CC_FLAGS=-std=c++17 -O3 -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5_cpp -lhdf5
SRC_DIR=./src/
BUILD_DIR=./build/
SRCS=$(shell find $(SRC_DIR) -type f -name "*.cpp")
OBJS=$(patsubst $(SRC_DIR)%.cpp, $(BUILD_DIR)/%.o,$(SRCS))
TARGET_FILE=main
DEPS := $(wildcard include/*.h)

INCLUDES=\
-I./include/ \
-I//usr/include/hdf5/serial/ \



$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(DEPS)
	@mkdir -p $(@D)
	$(CC) $(CC_FLAGS) $(INCLUDES) -c $< -o $@


$(BUILD_DIR)/$(TARGET_FILE): $(OBJS)
	$(CC) $^ -o $@ $(CC_FLAGS)
	@chmod +x $@


all: $(BUILD_DIR)/$(TARGET_FILE)
	chmod +x $^

run: all
	$(BUILD_DIR)/main

clean:
	rm build/*