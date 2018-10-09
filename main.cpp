#include <iostream>
#include <cassert>
#include "ndarray.hpp"




// ============================================================================
int main()
{
    assert(ndarray<3>(2, 3, 4).size() == 24);
    assert(ndarray<3>(2, 3, 4).shape() == (std::array<size_t, 3>{2, 3, 4}));
    assert(ndarray<3>(2, 3, 4)[0].shape() == (std::array<size_t, 2>{3, 4}));
}
