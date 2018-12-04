#include <iostream>
#include <fstream>
#include "include/ndarray.hpp"




int main()
{
    auto _ = nd::axis::all();
    // auto S = nd::selector<2>(3, 4);

    nd::ndarray<double, 1> A(10);

    nd::selector<2> sel({10, 10}, {0, 0}, {12, 12}, {2, 2});

    // std::cout << A.reshape(10, 1).select(_, _|0|1|0)(0, 0) << std::endl;

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
    auto D = C.select(_);
    std::cout << "This should be size 10, but I don't know why it works! " << D.size() << std::endl;

    std::ofstream("float64-345.bin") << nd::ndarray<double, 3>(3, 4, 5).dumps();
    std::ofstream("int32-88.bin") << nd::arange<int>(64).reshape(8, 8).dumps();

    auto I = nd::arange<int>(10).reshape(2, 5);

    for (auto x : I[0])
    {
        std::cout << x << std::endl;
    }

    return 0;
}
