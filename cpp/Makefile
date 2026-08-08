CXX := clang++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -Iinclude -pthread

.PHONY: all bench test mt-bench ring-test clean

all: bench test mt-bench ring-test

bench:
	mkdir -p bin
	$(CXX) $(CXXFLAGS) src/order_book.cpp src/main.cpp -o bin/bench

test:
	mkdir -p bin
	$(CXX) $(CXXFLAGS) src/order_book.cpp tests/order_book_test.cpp -o bin/order_book_test
	./bin/order_book_test

ring-test:
	mkdir -p bin
	$(CXX) $(CXXFLAGS) tests/ring_buffer_test.cpp -o bin/ring_buffer_test
	./bin/ring_buffer_test

mt-bench:
	mkdir -p bin
	$(CXX) $(CXXFLAGS) src/order_book.cpp src/mt_bench.cpp -o bin/mt_bench

clean:
	rm -rf bin
