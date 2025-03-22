CC      = gcc
CFLAGS  = -Iinclude -Wall -Wextra -std=c99
LDFLAGS = -lm

SRCS    = $(wildcard src/*.c)
TARGET  = tetris

# Determine operating system
ifeq ($(OS),Windows_NT)
    SHARED_EXT = .dll
    SHARED_FLAGS = -shared -DTETRIS_EXPORTS
    RM = del /Q
else
    SHARED_EXT = .so
    SHARED_FLAGS = -shared -fPIC
    RM = rm -f
endif

API_TARGET = api$(SHARED_EXT)

all: $(TARGET) $(API_TARGET)

$(TARGET): $(SRCS)
    $(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

$(API_TARGET): $(SRCS)
    $(CC) $(CFLAGS) $(SHARED_FLAGS) $(SRCS) -o $(API_TARGET) $(LDFLAGS)

clean:
    $(RM) $(TARGET) $(API_TARGET)

.PHONY: all clean