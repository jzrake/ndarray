CXXFLAGS = -std=c++14

default: test

test.o: selector.hpp ndarray.hpp

test: test.o catch.o
	$(CXX) -o $@ $(CXXFLAGS) $^

clean:
	$(RM) *.o test
