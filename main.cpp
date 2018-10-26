#include <iostream>
#include "ndarray.hpp"


int main()
{
    ndarray<1> A(10);

    for (int i = 0; i < A.size(); ++i)
    {
        A(i) = i;
    }

    auto str = A.dumps();
    auto B = ndarray<1>::loads(str);

    for (int i = 0; i < B.size(); ++i)
    {
        // std::cout << B(i, 0) << std::endl;
    }
    return 0;
}
