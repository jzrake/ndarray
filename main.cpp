#include <iostream>
#include "include/ndarray.hpp"




int main()
{
    auto _ = nd::axis::all();
    auto arr = nd::ndarray<double, 3>(10, 10, 10);
    auto n = arr.select(int(_|0), int(_|0), int(_|0));
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
    auto B = nd::ndarray<double, 1>::loads(str);

    for (int i = 0; i < B.size(); ++i)
    {
        std::cout << B(i) << std::endl;
    }

    auto C = (A + A) * nd::ones<int>(10);
    return 0;
}
