#include <iostream>
#include <cassert>
#include "ndarray.hpp"




// ============================================================================
int main()
{
    assert(ndarray<3>(2, 3, 4).size() == 24);
    assert(ndarray<3>(2, 3, 4).shape() == (std::array<size_t, 3>{2, 3, 4}));

    assert(ndarray<2>(3, 3)[0].shape() == (std::array<size_t, 1>{3}));
    assert(ndarray<2>(3, 3)[1].shape() == (std::array<size_t, 1>{3}));
    assert(ndarray<2>(3, 3)[1].strides() == (std::array<size_t, 1>{1}));

    assert(ndarray<3>(3, 4, 5)[0].shape() == (std::array<size_t, 2>{4, 5}));
    assert(ndarray<3>(3, 4, 5)[1].shape() == (std::array<size_t, 2>{4, 5}));
    assert(ndarray<3>(3, 4, 5)[0].strides() == (std::array<size_t, 2>{5, 1}));

    assert(ndarray<3>(2, 3, 4).offset(0, 0, 0) == 0);
    assert(ndarray<3>(2, 3, 4).offset(0, 0, 1) == 1);
    assert(ndarray<3>(2, 3, 4).offset(0, 1, 0) == 4);
    assert(ndarray<3>(2, 3, 4).offset(0, 2, 1) == 9);
    assert(ndarray<3>(2, 3, 4).offset(1, 2, 1) == 21);
}
