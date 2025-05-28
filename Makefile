CXX = g++
CXXFLAGS = -std=c++20 -Iinclude -O3 -Wall -Wextra -pedantic -flto -g -march=native
SRC := $(shell find src -name '*.cpp')
OBJ := $(SRC:.cpp=.o)
MAIN_TARGET = NN-ab-ovo
OPT_TARGET = NN-ab-ovo-opt
XOR_TARGET = NN-xor
MNIST_TARGET = NN-mnist

all: $(MAIN_TARGET) $(OPT_TARGET) $(XOR_TARGET) $(MNIST_TARGET)

$(MAIN_TARGET): $(OBJ) tests/main.o
	$(CXX) $(CXXFLAGS) -o $(MAIN_TARGET) $(OBJ) tests/main.o

$(OPT_TARGET): $(OBJ) tests/optimizations/optimizations.o
	$(CXX) $(CXXFLAGS) -o $(OPT_TARGET) $(OBJ) tests/optimizations/optimizations.o

$(XOR_TARGET): $(OBJ) tests/XOR/xor.o
	$(CXX) $(CXXFLAGS) -o $(XOR_TARGET) $(OBJ) tests/XOR/xor.o

$(MNIST_TARGET): $(OBJ) tests/MNIST/mnist.o
	$(CXX) $(CXXFLAGS) -o $(MNIST_TARGET) $(OBJ) tests/MNIST/mnist.o

tests/main.o: tests/main.cpp
	$(CXX) $(CXXFLAGS) -c tests/main.cpp -o tests/main.o

tests/optimizations.o: tests/optimizations/optimizations.cpp
	$(CXX) $(CXXFLAGS) -c tests/optimizations/optimizations.cpp -o tests/optimizations/optimizations.o

tests/XOR/xor.o: tests/XOR/xor.cpp
	$(CXX) $(CXXFLAGS) -c tests/XOR/xor.cpp -o tests/XOR/xor.o

tests/MNIST/mnist.o: tests/MNIST/mnist.cpp
	$(CXX) $(CXXFLAGS) -c tests/MNIST/mnist.cpp -o tests/MNIST/mnist.o

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) tests/*.o tests/XOR/*.o tests/MNIST/*.o $(MAIN_TARGET) $(OPT_TARGET) $(XOR_TARGET) $(MNIST_TARGET)