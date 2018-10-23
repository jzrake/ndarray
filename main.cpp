#include <iostream>
#include <cassert>
#include "ndarray.hpp"




// ============================================================================
void test_selector()
{
    auto S = selector<4, 0> {{4, 3, 2, 3}, {0, 0, 0, 0}, {4, 3, 2, 3}};

    assert(S.count == (std::array<int, 4>{4, 3, 2, 3}));
    assert(S.axis == 0);
    assert(S.within(std::make_tuple(0, 1)).axis == 1);
    assert(S.within(std::make_tuple(0, 1)).size() == 18);
    assert(S.within(std::make_tuple(0, 1), std::make_tuple(0, 1)).size() == 6);
    assert(S.within(std::make_tuple(0, 1), std::make_tuple(0, 1), std::make_tuple(0, 1)).size() == 3);
    assert(S.within(std::make_tuple(0, 1), std::make_tuple(0, 1), std::make_tuple(0, 1), std::make_tuple(0, 1)).size() == 1);
    assert(S.within(std::make_tuple(0, 1), std::make_tuple(0, 1), std::make_tuple(0, 1)).axis == 3);
    assert(S.within(std::make_tuple(0, 1), std::make_tuple(0, 1), 1, std::make_tuple(0, 1)).rank == 3);
    assert(S.within(std::make_tuple(0, 1), std::make_tuple(0, 1), 1, std::make_tuple(0, 1)).size() == 1);
    assert(S.collapse(0).count.size() == 3);
}


void test_ndarray()
{
    {
        // auto A = ndarray<1>(2);
        // std::cout << &A(1) - &A(0) << std::endl;
        // std::cout << &A[1]() - &A(0) << std::endl;
    }
    {
        auto A = ndarray<2>(3, 2);
        std::cout << &A(1, 1) - &A(0, 0) << std::endl;
        std::cout << &A[1](1) - &A(0, 0) << std::endl;
        std::cout << &A[1][1]() - &A(0, 0) << std::endl;
    }
    {
        auto A = ndarray<5>(3, 4, 5, 6, 7);
        assert(&A(0, 0, 0, 0, 1) - &A(0, 0, 0, 0, 0) == 1);
        assert(&A[0](0, 0, 0, 1) - &A(0, 0, 0, 0, 0) == 1);
        assert(&A[0][0](0, 0, 1) - &A(0, 0, 0, 0, 0) == 1);
        assert(&A[0][0][0](0, 1) - &A(0, 0, 0, 0, 0) == 1);
        assert(&A[0][0][0][0](1) - &A(0, 0, 0, 0, 0) == 1);
        assert(&A[0][0][0][0][1]() - &A(0, 0, 0, 0, 0) == 1);

        // std::cout << &A(1, 1, 1, 1, 1) - &A(0, 0, 0, 0, 0) << std::endl;
        // std::cout << &A[1](1, 1, 1, 1) - &A(0, 0, 0, 0, 0) << std::endl;
        // std::cout << &A[1][1](1, 1, 1) - &A(0, 0, 0, 0, 0) << std::endl;
        // std::cout << &A[1][1][1](1, 1) - &A(0, 0, 0, 0, 0) << std::endl;
        // std::cout << &A[1][1][1][1](1) - &A(0, 0, 0, 0, 0) << std::endl;
        // std::cout << &A[1][1][1][1][1]() - &A(0, 0, 0, 0, 0) << std::endl;

        // assert(A.size() == 2520);
    }
    {
        auto A = ndarray<5>(3, 4, 5, 6, 7);
        A(0, 0, 0, 0, 0) = 2;
        A(1, 0, 2, 3, 4) = 10234;
        A(2, 1, 0, 2, 5) = 21025;

        assert(A(1, 0, 2, 3, 4) == 10234);
        assert(A(2, 1, 0, 2, 5) == 21025);
    }
    {
        auto A = ndarray<0>();
    }
}




// ============================================================================
int main()
{
    test_selector();
    test_ndarray();

 //    assert(ndarray<3>(2, 3, 4).size() == 24);
 //    assert(ndarray<3>(2, 3, 4).shape() == (std::array<int, 3>{2, 3, 4}));

 //    assert(ndarray<2>(3, 3)[0].shape() == (std::array<int, 1>{3}));
 //    assert(ndarray<2>(3, 3)[1].shape() == (std::array<int, 1>{3}));
 //    assert(ndarray<2>(3, 3)[1].strides() == (std::array<int, 1>{1}));

 //    assert(ndarray<3>(3, 4, 5)[0].shape() == (std::array<int, 2>{4, 5}));
 //    assert(ndarray<3>(3, 4, 5)[1].shape() == (std::array<int, 2>{4, 5}));
 //    assert(ndarray<3>(3, 4, 5)[0].strides() == (std::array<int, 2>{5, 1}));

 //    assert(ndarray<3>(2, 3, 4).offset(0, 0, 0) == 0);
 //    assert(ndarray<3>(2, 3, 4).offset(0, 0, 1) == 1);
 //    assert(ndarray<3>(2, 3, 4).offset(0, 1, 0) == 4);
 //    assert(ndarray<3>(2, 3, 4).offset(0, 2, 1) == 9);
 //    assert(ndarray<3>(2, 3, 4).offset(1, 2, 1) == 21);

	// ndarray<3> A(3, 3, 3);
	// ndarray<3> B(3, 3, 3);
 //    assert(A.shares(A) == true);
 //    assert(A.shares(B) == false);
 //    assert(A.shares(A.reshape(9, 3)) == true);

 //    A(0, 0, 0) = 1;
 //    A(0, 1, 0) = 2;
 //    A(0, 1, 2) = 3;

 //    auto C = A[0];

 //    assert(A(0, 0, 0) == 1);
 //    assert(A(0, 1, 0) == 2);
 //    assert(A(0, 1, 2) == 3);
 //    assert(A.select(1, 1, 3).shape() == (std::array<int, 3>{3, 2, 3}));
 //    assert(A.select(1, 1, 3)(0, 0, 0) == 2);
 //    assert(A.select(1, 1, 3)(0, 0, 2) == 3);

 //    assert(A.within(std::make_tuple(0, 3), std::make_tuple(1, 3), std::make_tuple(2, 3)).shape() == (std::array<int, 3>{3, 2, 1}));
 //    assert(A.within(std::make_tuple(0, 2), std::make_tuple(1, 3), std::make_tuple(0, 3)).shape() == (std::array<int, 3>{2, 2, 3}));
}
