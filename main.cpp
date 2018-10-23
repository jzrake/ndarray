#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "ndarray.hpp"




// ============================================================================
TEST_CASE("selector<4> passes basic sanity checks", "[selector]")
{
    auto S = selector<4>{{4, 3, 2, 3}, {0, 0, 0, 0}, {4, 3, 2, 3}};

    REQUIRE(S.axis == 0);
    REQUIRE(S.count == std::array<int, 4>{4, 3, 2, 3});
    REQUIRE(S.shape() == S.count);
}


TEST_CASE("selector<1> works when instantiated as a subset", "[selector]")
{
    auto S = selector<1>{{10}, {2}, {8}};

    REQUIRE(S.axis == 0);
    REQUIRE(S.count == std::array<int, 1>{10});
    REQUIRE(S.shape() == std::array<int, 1>{6});
}


TEST_CASE("selector<2> works when instantiated as a subset", "[selector]")
{
    auto S = selector<2>{{10, 12}, {2, 4}, {8, 6}};

    REQUIRE(S.axis == 0);
    REQUIRE(S.count == std::array<int, 2>{10, 12});
    REQUIRE(S.shape() == std::array<int, 2>{6, 2});
    REQUIRE(S.size() == 12);
}


TEST_CASE("selector<2> collapses properly to selector<1>", "[selector::collapse]")
{
    auto S = selector<2>{{10, 12}, {0, 0}, {10, 12}};

    REQUIRE(S.collapse(0).axis == 0);
    REQUIRE(S.collapse(0).count == std::array<int, 1>{120});
    REQUIRE(S.collapse(0).shape() == std::array<int, 1>{12});
    REQUIRE(S.collapse(0).size() == 12);
}


TEST_CASE("selector<2> subset collapses properly to selector<1>", "[selector::collapse]")
{
    auto S = selector<2>{{10, 12}, {2, 4}, {8, 6}};

    REQUIRE(S.collapse(0).axis == 0);
    REQUIRE(S.collapse(0).count == std::array<int, 1>{120});
    REQUIRE(S.collapse(0).start == std::array<int, 1>{24});
    REQUIRE(S.collapse(0).final == std::array<int, 1>{26});
    REQUIRE(S.collapse(0).size() == 2);
}


TEST_CASE("selector<2> subset is created properly from a selector<2>", "selector::select")
{
    auto S = selector<2>{{10, 12}, {0, 0}, {10, 12}};
    REQUIRE(S.select(std::make_tuple(0, 10)).reset() == S);
    REQUIRE(S.select(std::make_tuple(2, 4)).reset() == selector<2>{{10, 12}, {2, 0}, {4, 12}});
    REQUIRE(S.select(std::make_tuple(2, 8)).reset().select(std::make_tuple(2, 4)).reset() == selector<2>{{10, 12}, {4, 0}, {6, 12}});
}


TEST_CASE("selector<1> next advances properly", "[selector::next]")
{
    auto S = selector<1>{{10}, {0}, {10}};
    auto I = std::array<int, 1>{0};
    auto i = 0;

    do {
        REQUIRE(i == I[0]);
        ++i;
    } while (S.next(I));
}


TEST_CASE("selector<2> next advances properly", "[selector::next]")
{
    auto S = selector<2>{{10, 10}, {0, 0}, {10, 10}};
    auto I = std::array<int, 2>{0, 0};
    auto i = 0;
    auto j = 0;

    do {
        REQUIRE(i == I[0]);
        REQUIRE(j == I[1]);

        if (++j == 10)
        {
            j = 0;
            ++i;
        }
    } while (S.next(I));
}


TEST_CASE("selector<2> subset next advances properly", "[selector::next]")
{
    auto S = selector<2>{{10, 10}, {2, 4}, {8, 6}};
    auto I = std::array<int, 2>{2, 4};
    auto i = 2;
    auto j = 4;

    do {
        REQUIRE(i == I[0]);
        REQUIRE(j == I[1]);

        if (++j == 6)
        {
            j = 4;
            ++i;
        }
    } while (S.next(I));
}


TEST_CASE("selector<2> subset iterator passes sanity checks", "[selector::iterator]")
{
    auto S = selector<2>{{10, 10}, {2, 4}, {8, 6}};
    auto I = S.start;

    for (auto index : S)
    {
        REQUIRE(index == I);
        S.next(I);
    }
}


TEST_CASE("scalar ndarray (ndarray<0>) passes sanity checks", "ndarray")
{
    auto A = ndarray<0>(3.14);
    REQUIRE(A.rank == 0);
    REQUIRE(A() == 3.14);
    REQUIRE(A == 3.14);

    A() = 2.0;

    REQUIRE(A() == 2.0);
    REQUIRE(A == 2.0);
}


TEST_CASE("ndarray<1> passes sanity checks", "ndarray")
{
    auto A = ndarray<1>{0, 1, 2, 3, 4};
    REQUIRE(A.rank == 1);
    REQUIRE(A.size() == 5);
    REQUIRE(A.shape() == std::array<int, 1>{5});
    REQUIRE(A(0) == 0);
    REQUIRE(A(4) == 4);
    REQUIRE(A[0] == 0);
    REQUIRE(A[4] == 4);
    REQUIRE(A.is(A));
    REQUIRE(! A.copy().is(A));
}


TEST_CASE("ndarray<1> iterator passes sanity checks", "ndarray::iterator")
{
    auto A = ndarray<1>{0, 1, 2, 3, 4};
    auto x = 0.0;

    REQUIRE(A.begin() == A.begin());
    REQUIRE(A.begin() != A.end());

    SECTION("conventional iterator works")
    {
        for (auto it = A.begin(); it != A.end(); ++it)
        {
            REQUIRE(*it == x++);
        }
    }
    SECTION("range-based for loop works")
    {
        for (auto y : A)
        {
            REQUIRE(y == x++);
        }
    }
}


TEST_CASE("ndarray<1> can be sliced, copied, and compared", "ndarray::select")
{
    auto A = ndarray<1>{0, 1, 2, 3, 4};
    auto B = ndarray<1>{0, 1, 2, 3};
    REQUIRE(B.container() == A.select(std::make_tuple(0, 4)).copy().container());
}


TEST_CASE("ndarray<2> can be sliced, copied, and compared", "ndarray::select")
{
    auto A = ndarray<2>(3, 4);
    auto B = A.select(std::make_tuple(0, 2));

    for (int i = 0; i < A.shape()[0]; ++i)
    {
        for (int j = 0; j < A.shape()[1]; ++j)
        {
            A(i, j) = i + j;
        }
    }

    for (int i = 0; i < B.shape()[0]; ++i)
    {
        for (int j = 0; j < B.shape()[1]; ++j)
        {
            REQUIRE(A(i, j) == B(i, j));
        }
    }
}
