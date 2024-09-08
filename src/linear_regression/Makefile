


CC=g++
CC_FLAGS=-std=c++17 -O3
SRC_DIR=./
BUILD_DIR=./
SRCS=$(shell find $(SRC_DIR) -type f -name "*.cpp")
OBJS=$(patsubst $(SRC_DIR)%.cpp, $(BUILD_DIR)/%.o,$(SRCS))
TARGET_FILE=linear



$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $
	@mkdir -p $(@D)
	$(CC) $(CC_FLAGS) $(INCLUDES) -c $< -o $@


$(BUILD_DIR)/$(TARGET_FILE): $(OBJS)
	$(CC) $^ -o $@ $(CC_FLAGS)
	@chmod +x $@


all: $(BUILD_DIR)/$(TARGET_FILE)
	chmod +x $^

run: all
	./linear

clean:
	rm model.o util.o main.o