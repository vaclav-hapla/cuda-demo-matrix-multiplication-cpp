#### CONFIGURATION VARIABLES ####

# Define standard directories
SRC_DIR := src
BUILD_DIR := build
LIB_DIR := lib
INCLUDE_DIR := include
BIN_DIR := bin
TEST_DIR := tests

# Define the output shared library name
LIB_NAME := cudamatmult

# Define the compilers and flags
CC := gcc
NVCC := nvcc
CPPFLAGS := -I$(INCLUDE_DIR)
# TODO: add optimization flags, including -DNDEBUG, conditionally
CFLAGS := -g -O1 -std=c11 -fPIC $(CPPFLAGS)
NVCCFLAGS := -g -O1 --gpu-architecture=sm_89 -Xcompiler -fPIC $(CPPFLAGS)
LDFLAGS := -shared
LIBS := -L$(LIB_DIR) -Wl,-rpath,$(LIB_DIR) -l$(LIB_NAME) -L/usr/local/cuda/lib64 -lcudart


#### INTERNAL VARIABLES ####

# Define the output library
OUTPUT_LIB := $(LIB_DIR)/lib$(LIB_NAME).so

# Find all .c and .cu files in SRC_DIR
SRCS_C := $(wildcard $(SRC_DIR)/*.c)
SRCS_CU := $(wildcard $(SRC_DIR)/*.cu)
SRCS := $(SRCS_C) $(SRCS_CU)

# Convert source files to object files in the build directory
OBJS := $(SRCS_C:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
OBJS += $(SRCS_CU:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Find all test source files in TEST_DIR
TEST_SRCS := $(wildcard $(TEST_DIR)/test*.c)
TEST_OBJS := $(TEST_SRCS:$(TEST_DIR)/%.c=$(BUILD_DIR)/$(TEST_DIR)/%.o)
TEST_EXES := $(TEST_SRCS:$(TEST_DIR)/%.c=$(BIN_DIR)/$(TEST_DIR)/%)


#### COMMANDS ####

# Default target
all: lib build_test

# Library target
lib: $(OUTPUT_LIB)

# Build tests target
build_test: lib $(TEST_EXES)

# Target to run all tests
test: build_test
	@$(MAKE) --no-print-directory $(TEST_EXES:=.run)

# Clean target
clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR) $(BIN_DIR)

print:
	@echo CC=$(CC)
	@echo NVCC=$(NVCC)
	@echo CFLAGS=$(CFLAGS)
	@echo NVCCFLAGS=$(NVCCFLAGS)
	@echo LDFLAGS=$(LDFLAGS)
	@echo
	@echo SRC_DIR=$(SRC_DIR)
	@echo BUILD_DIR=$(BUILD_DIR)
	@echo LIB_DIR=$(LIB_DIR)
	@echo INCLUDE_DIR=$(INCLUDE_DIR)
	@echo BIN_DIR=$(BIN_DIR)
	@echo TEST_DIR=$(TEST_DIR)
	@echo
	@echo SRCS=$(SRCS)
	@echo OBJS=$(OBJS)
	@echo
	@echo TEST_SRCS=$(TEST_SRCS)
	@echo TEST_OBJS=$(TEST_OBJS)
	@echo TEST_EXES=$(TEST_EXES)
	@echo
	@echo OUTPUT_LIB=$(OUTPUT_LIB)

.PHONY: all clean lib build_test test print


#### RULES ####

# Rule to build the shared library
$(OUTPUT_LIB): $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(LDFLAGS) -o $@ $(OBJS)

# Rule to compile .c files into the build directory
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to compile .cu files into the build directory
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Rule to compile test .c files into the build directory
$(BUILD_DIR)/$(TEST_DIR)/%.o: $(TEST_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to link test executables
$(BIN_DIR)/$(TEST_DIR)/%: $(BUILD_DIR)/$(TEST_DIR)/%.o $(OUTPUT_LIB)
	@mkdir -p $(dir $@)
	$(CC) -o $@ $< $(LIBS)

# Pattern rule to run individual test executables
.PHONY: %.run
$(BIN_DIR)/$(TEST_DIR)/%.run: $(BIN_DIR)/$(TEST_DIR)/%
	@echo "Running $<"
	@$<
