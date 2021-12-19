BUILD_PROFILE = debug
OUT_DIR = target/$(BUILD_PROFILE)

CC = clang

AFL_URL = https://github.com/AFLplusplus/AFLplusplus
AFL_DIR = AFLplusplus
AFL_CC = $(AFL_DIR)/afl-cc

EXAMPLE_DIR = examples/forkserver_simple
EXAMPLE_TARGET = $(EXAMPLE_DIR)/src/program.c


aflplusplus:
	@if [ ! -d $(AFL_DIR) ]; then \
		git clone $(AFL_URL) -o $(AFL_DIR); \
		cd $(AFL_DIR); \
		make all; \
	fi


afl: aflplusplus $(EXAMPLE_TARGET)
	# compile target program
	$(AFL_CC) $(EXAMPLE_TARGET) -o $(OUT_DIR)/program


clean:
	cargo clean
	rm -rf $(AFL_DIR)


.PHONY: checks
