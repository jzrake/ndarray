CXXFLAGS = -std=c++14

default: main sel

main: main.o catch_main.o
	$(CXX) -o $@ $(CXXFLAGS) $^

sel: sel.o catch_main.o
	$(CXX) -o $@ $(CXXFLAGS) $^
