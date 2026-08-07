CXX := clang++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -Iinclude

.PHONY: all bench test clean

all: bench test

bench:
	mkdir -p bin
	$(CXX) $(CXXFLAGS) src/order_book.cpp src/main.cpp -o bin/bench

test:
	mkdir -p bin
	$(CXX) $(CXXFLAGS) src/order_book.cpp tests/order_book_test.cpp -o bin/order_book_test
	./bin/order_book_test

clean:
	rm -rf bin
