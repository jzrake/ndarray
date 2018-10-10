CXXFLAGS = -std=c++14

main : main.cpp ndarray.hpp
	$(CXX) -o $@ $(CXXFLAGS) $<