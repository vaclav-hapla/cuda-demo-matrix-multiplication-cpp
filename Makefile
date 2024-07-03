#### CONFIGURATION VARIABLES ####

# Define standard directories
SRC_DIR := src
BUILD_DIR := build
LIB_DIR := lib
INCLUDE_DIR := include
BIN_DIR := bin
TEST_DIR := tests

# Define the output shared library name
LIB_NAME := cudamatmult++

# Define the compilers and flags
CXX := g++
NVCC := nvcc
CPPFLAGS := -I$(INCLUDE_DIR)
# TODO: add optimization flags, including -DNDEBUG, conditionally
CXXFLAGS := -g -O0 -std=c++20 -fPIC -MMD -MP $(CPPFLAGS)
NVCCFLAGS := -dc -g -G -O0 --gpu-architecture=sm_89 -Xcompiler -fPIC --std=c++20 $(CPPFLAGS)
LDFLAGS := --gpu-architecture=sm_89
LIBS := -L$(LIB_DIR) -Xlinker -rpath -Xlinker $(PWD)/$(LIB_DIR) -l$(LIB_NAME) -L/usr/local/cuda/lib64 -lcudart


#### INTERNAL VARIABLES ####

# Define the output library
OUTPUT_LIB := $(LIB_DIR)/lib$(LIB_NAME).so

# Find all .c and .cu files in SRC_DIR
SRCS_CXX := $(wildcard $(SRC_DIR)/*.cpp)
SRCS_CU := $(wildcard $(SRC_DIR)/*.cu)
SRCS := $(SRCS_CXX) $(SRCS_CU)

# Convert source files to object files in the build directory
OBJS := $(SRCS_CXX:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
OBJS += $(SRCS_CU:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Find all dependency files
DEPS = $(OBJS:.o=.d)

# Find all test source files in TEST_DIR
TEST_SRCS_CXX := $(wildcard $(TEST_DIR)/test*.cpp)
TEST_SRCS_CU := $(wildcard $(TEST_DIR)/test*.cu)
TEST_SRCS := $(TEST_SRCS_CXX) $(TEST_SRCS_CU)
TEST_OBJS := $(TEST_SRCS_CXX:$(TEST_DIR)/%.cpp=$(BUILD_DIR)/$(TEST_DIR)/%.o)
TEST_OBJS += $(TEST_SRCS_CU:$(TEST_DIR)/%.cu=$(BUILD_DIR)/$(TEST_DIR)/%.o)
TEST_EXES := $(TEST_OBJS:$(BUILD_DIR)/$(TEST_DIR)/%.o=$(BIN_DIR)/$(TEST_DIR)/%)


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
	@echo CXX=$(CXX)
	@echo NVCC=$(NVCC)
	@echo CXXFLAGS=$(CXXFLAGS)
	@echo NVCCFLAGS=$(NVCCFLAGS)
	@echo LDFLAGS=$(LDFLAGS)
	@echo LIBS=$(LIBS)
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

# Include all dependency files
-include $(DEPS)

# Rule to build the shared library
$(OUTPUT_LIB): $(OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) -shared $(LDFLAGS) -o $@ $(OBJS)

# Rule to compile .cpp files into the build directory
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile .cu files into the build directory
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Rule to compile test .cpp files into the build directory
$(BUILD_DIR)/$(TEST_DIR)/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile test .cu files into the build directory
$(BUILD_DIR)/$(TEST_DIR)/%.o: $(TEST_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Rule to link test executables
$(BIN_DIR)/$(TEST_DIR)/%: $(BUILD_DIR)/$(TEST_DIR)/%.o $(OUTPUT_LIB)
	@mkdir -p $(dir $@)
	$(NVCC) $(LDFLAGS) -o $@ $< $(LIBS)

# Pattern rule to run individual test executables
.PHONY: %.run
$(BIN_DIR)/$(TEST_DIR)/%.run: $(BIN_DIR)/$(TEST_DIR)/%
	@echo "Running $<"
	@$<
