CC = gcc
CFLAGS = -Wall -Wextra -Werror -O2 -fPIC -std=c11
INCLUDES = -Iinclude
LDFLAGS = -shared -lm

SRCDIR = src
OBJDIR = build
TARGET = liborbit_wars.so

SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -rf $(OBJDIR) $(TARGET)

test: $(TARGET)
	cd .. && .venv/bin/python -m pytest c_engine/tests/test_parity.py -v
