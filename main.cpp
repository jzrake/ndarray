#include <iostream>
#include <cassert>
#include "ndarray.hpp"




// ============================================================================
int main()
{
    assert(ndarray<3>(2, 3, 4).size() == 24);
    assert(ndarray<3>(2, 3, 4).shape() == (std::array<int, 3>{2, 3, 4}));

    assert(ndarray<2>(3, 3)[0].shape() == (std::array<int, 1>{3}));
    assert(ndarray<2>(3, 3)[1].shape() == (std::array<int, 1>{3}));
    assert(ndarray<2>(3, 3)[1].strides() == (std::array<int, 1>{1}));

    assert(ndarray<3>(3, 4, 5)[0].shape() == (std::array<int, 2>{4, 5}));
    assert(ndarray<3>(3, 4, 5)[1].shape() == (std::array<int, 2>{4, 5}));
    assert(ndarray<3>(3, 4, 5)[0].strides() == (std::array<int, 2>{5, 1}));

    assert(ndarray<3>(2, 3, 4).offset(0, 0, 0) == 0);
    assert(ndarray<3>(2, 3, 4).offset(0, 0, 1) == 1);
    assert(ndarray<3>(2, 3, 4).offset(0, 1, 0) == 4);
    assert(ndarray<3>(2, 3, 4).offset(0, 2, 1) == 9);
    assert(ndarray<3>(2, 3, 4).offset(1, 2, 1) == 21);


	ndarray<3> A(3, 3, 3);
	ndarray<3> B(3, 3, 3);
    assert(A.shares_memory_with(A) == true);
    assert(A.shares_memory_with(B) == false);
    assert(A.shares_memory_with(A.reshape(9, 3)) == true);
}
