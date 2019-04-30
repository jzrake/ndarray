CXXFLAGS = -std=c++17 -O0 -Wextra -Wno-missing-braces
HEADERS = ndarray.hpp

default: test main

main.o: ndarray.hpp

test.o: $(HEADERS)

test: test.o catch.o
	$(CXX) -o $@ $(CXXFLAGS) $^

main: main.o other.o
	$(CXX) -o $@ $(CXXFLAGS) $^

clean:
	$(RM) *.o test main
