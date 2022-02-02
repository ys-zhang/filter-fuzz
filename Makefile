BUILD_PROFILE = debug
OUT_DIR = target/$(BUILD_PROFILE)

CC = clang
LLVM_CONFIG = llvm-config-12

AFL_URL = https://github.com/AFLplusplus/AFLplusplus
AFL_DIR = AFLplusplus
AFL_CC = $(AFL_DIR)/afl-cc

EXAMPLE_DIR = examples/forkserver_simple
EXAMPLE_TARGET = $(EXAMPLE_DIR)/src/program.c


forkserver-simple: afl
	RUST_BACKTRACE=1 cargo run --example forkserver_simple -- $(OUT_DIR)/program $(EXAMPLE_DIR)/corpus

afl: aflplusplus $(EXAMPLE_TARGET)
	@# compile target program
	$(AFL_CC) $(EXAMPLE_TARGET) -o $(OUT_DIR)/program


aflplusplus:
	@if [ ! -d $(AFL_DIR) ]; then \
		git clone $(AFL_URL) -o $(AFL_DIR); \
		cd $(AFL_DIR); \
		export LLVM_CONFIG=$(LLVM_CONFIG); \
		make all; \
	fi


model-dense:
	@rm -rf models/test-model
	python3 src/py/create_model.py -m dense -i 512 -o 2048 -n test-model

clean:
	cargo clean
	rm -rf $(AFL_DIR)


.PHONY: checks
