CXX = g++
CXXFLAGS = -std=c++11 -Iinclude
SRC = src/Matrix.cpp src/DenseLayer.cpp src/Sequential.cpp src/Loss.cpp src/Activation.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = NN-ab-ovo

all: $(TARGET)

$(TARGET): $(OBJ) tests/main.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ) tests/main.o

tests/main.o: tests/main.cpp
	$(CXX) $(CXXFLAGS) -c tests/main.cpp -o tests/main.o

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) tests/*.o $(TARGET)
