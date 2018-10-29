#include <iostream>
#include "ndarray.hpp"




int main()
{
    auto S = nd::selector<2>(3, 4);
    auto I = std::array<int, 2>{0, 0};

    while (S.next(I))
    {
        std::cout << I[0] << " " << I[1] << std::endl;
    }

    nd::ndarray<double, 1> A(10);

    for (int i = 0; i < A.size(); ++i)
    {
        A(i) = i;
    }

    auto str = A.dumps();
    // auto B = nd::ndarray<double, 1>::loads(str);

    // for (int i = 0; i < B.size(); ++i)
    {
        // std::cout << B(i, 0) << std::endl;
    }
    return 0;
}
