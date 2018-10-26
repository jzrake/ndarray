CXXFLAGS = -std=c++14

default: test main

test.o: selector.hpp ndarray.hpp
main.o: selector.hpp ndarray.hpp

test: test.o catch.o
	$(CXX) -o $@ $(CXXFLAGS) $^

main: main.o
	$(CXX) -o $@ $(CXXFLAGS) $^

clean:
	$(RM) *.o test
