CXXFLAGS = -std=c++14 -O3 -Wextra -Wno-missing-braces
HEADERS = selector.hpp shape.hpp buffer.hpp ndarray.hpp

default: test main

main.o: include/ndarray.hpp

test.o: $(HEADERS)

include/ndarray.hpp: $(HEADERS)
	./collate.sh $^ > $@

test: test.o catch.o
	$(CXX) -o $@ $(CXXFLAGS) $^

main: main.o other.o
	$(CXX) -o $@ $(CXXFLAGS) $^

clean:
	$(RM) *.o test main
