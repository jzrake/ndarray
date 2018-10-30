CXXFLAGS = -std=c++14

HEADERS = selector.hpp shape.hpp buffer.hpp ndarray.hpp

default: test main
main.o: include/ndarray.hpp
test.o: $(HEADERS)
include/ndarray.hpp: $(HEADERS)
	./collate.sh $^ > $@

test: test.o catch.o
	$(CXX) -o $@ $(CXXFLAGS) $^

main: main.o
	$(CXX) -o $@ $(CXXFLAGS) $^

clean:
	$(RM) *.o test main
