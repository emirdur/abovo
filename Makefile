CXX = g++
CXXFLAGS = -std=c++11 -Iinclude
SRC = src/Matrix.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = NN-ab-ovo

all: $(TARGET)

$(TARGET): $(OBJ) tests/main.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ) tests/main.o

tests/main.o: tests/main.cpp
	$(CXX) $(CXXFLAGS) -c tests/main.cpp -o tests/main.o

src/Matrix.o: src/Matrix.cpp
	$(CXX) $(CXXFLAGS) -c src/Matrix.cpp -o src/Matrix.o

clean:
	rm -f $(OBJ) tests/*.o $(TARGET)
