#include <iostream>
#include "ndarray.hpp"


int main()
{
    ndarray<2> A(10, 10);

    for (int i = 0; i < A.size(); ++i)
    {
        A(i, 0) = i;
    }

    auto str = A.dumps();
    auto B = ndarray<2>::loads(str);

    for (int i = 0; i < B.size(); ++i)
    {
        std::cout << B(i, 0) << std::endl;
    }
    return 0;
}
