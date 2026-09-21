# PC 端自测：把端侧算法核心（MFCC / 决策 / 量化）编译成原生可执行文件，
# 不依赖任何 ESP32 工具链，纯 C99 + libm。
#
#   make test
#
# dump_mfcc      : 导出一段 PCM 的 MFCC，供 numpy 交叉比对，并核对流式与整段路径一致
# test_decision  : 报警决策（阈值 + 连续确认）与 INT8 量化往返
#
CC      ?= cc
CFLAGS  ?= -std=c99 -Wall -Wextra -O2
FW      := ../../firmware/HomeGuard
INC     := -I$(FW)

.PHONY: all test clean

all: dump_mfcc test_decision

dump_mfcc: dump_mfcc.c $(FW)/mfcc.c $(FW)/fft.c
	$(CC) $(CFLAGS) $(INC) -o $@ $^ -lm

test_decision: test_decision.c $(FW)/decision.c
	$(CC) $(CFLAGS) $(INC) -o $@ $^ -lm

test: all
	python gen_test_pcm.py
	./dump_mfcc
	python compare_mfcc.py
	./test_decision

clean:
	rm -f dump_mfcc dump_mfcc.exe test_decision test_decision.exe mfcc_out.txt test_pcm.bin
