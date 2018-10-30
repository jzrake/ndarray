#include <iostream>
#include "include/ndarray.hpp"




int main()
{
    auto S = nd::selector<2>(3, 4);
    auto I = std::array<int, 2>{0, 0};

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

    auto _ = nd::axis::all();
    auto D = C.select(_);
    std::cout << "This should be size 10, but I don't know why it works! " << D.size() << std::endl;

    return 0;
}
