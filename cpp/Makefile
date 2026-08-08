CXX := clang++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -Iinclude -pthread

.PHONY: all bench test mt-bench ring-test vol-test vol-bench feed clean

all: bench test mt-bench ring-test vol-test vol-bench feed

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

vol-test:
	mkdir -p bin
	$(CXX) $(CXXFLAGS) src/volatility.cpp tests/volatility_test.cpp -o bin/volatility_test
	./bin/volatility_test

vol-bench:
	mkdir -p bin
	$(CXX) $(CXXFLAGS) src/volatility.cpp src/volatility_bench.cpp -o bin/volatility_bench

feed:
	mkdir -p bin
	$(CXX) $(CXXFLAGS) src/feed_publisher.cpp -o bin/feed_publisher
	$(CXX) $(CXXFLAGS) src/order_book.cpp src/feed_subscriber.cpp -o bin/feed_subscriber

clean:
	rm -rf bin
