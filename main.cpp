#include <iostream>
#include <cassert>
#include "ndarray.hpp"




// ============================================================================
template<int Rank, int Axis>
struct selector
{
    enum { rank = Rank, axis = Axis };

    selector<rank - 1, axis> collapse(int start_index) const
    {
        static_assert(rank > 0, "selector: cannot collapse zero-rank selector");
        static_assert(axis < rank, "selector: attempting to index on axis >= rank");

        std::array<int, rank - 1> _count;
        std::array<int, rank - 1> _start;
        std::array<int, rank - 1> _final;

        for (int n = 0; n < axis; ++n)
        {
            _count[n] = count[n];
            _start[n] = start[n];
            _final[n] = final[n];
        }

        for (int n = axis + 1; n < rank - 1; ++n)
        {
            _count[n] = count[n + 1];
            _start[n] = start[n + 1];
            _final[n] = final[n + 1];
        }

        _count[axis] = count[axis] * count[axis + 1];
        _start[axis] = count[axis] * start_index;
        _final[axis] = count[axis] * start_index + count[axis + 1];

        return {_count, _start, _final};
    }

    selector<rank - 1, axis> within(int start_index) const
    {
        return collapse(start_index);
    }

    selector<rank, axis + 1> within(std::tuple<int, int> range) const
    {
        static_assert(axis < rank, "selector: attempting to index on axis >= rank");

        auto _count = count;
        auto _start = start;
        auto _final = final;

        _start[axis] = start[axis] + std::get<0>(range);
        _final[axis] = start[axis] + std::get<1>(range);

        return {_count, _start, _final};
    }

    template<typename First, typename... Rest>
    auto within(First first, Rest... rest) const
    {
        return within(first).within(rest...);
    }

    std::array<int, rank> shape() const
    {
        std::array<int, rank> s;

        for (int n = 0; n < rank; ++n)
        {
            s[n] = final[n] - start[n];
        }
        return s;
    }

    int size() const
    {
        auto s = shape();
        return std::accumulate(s.begin(), s.end(), 1, std::multiplies<>());
    }

    std::array<int, rank> count;
    std::array<int, rank> start;
    std::array<int, rank> final;
};




// ============================================================================
void test_selector()
{
    auto S = selector<4, 0> {{4, 3, 2, 3}, {0, 0, 0, 0}, {4, 3, 2, 3}};

    assert(S.collapse(0).count.size() == 3);

    assert(S.count == (std::array<int, 4>{4, 3, 2, 3}));
    assert(S.axis == 0);
    assert(S.within(std::make_tuple(0, 1)).axis == 1);
    assert(S.within(std::make_tuple(0, 1), std::make_tuple(0, 1), std::make_tuple(0, 1)).axis == 3);

    assert(S.within(std::make_tuple(0, 1)).size() == 18);
    assert(S.within(std::make_tuple(0, 1), std::make_tuple(0, 1)).size() == 6);
    assert(S.within(std::make_tuple(0, 1), std::make_tuple(0, 1), std::make_tuple(0, 1)).size() == 3);
    assert(S.within(std::make_tuple(0, 1), std::make_tuple(0, 1), std::make_tuple(0, 1), std::make_tuple(0, 1)).size() == 1);

    assert(S.within(std::make_tuple(0, 1), std::make_tuple(0, 1), 1, std::make_tuple(0, 1)).rank == 3);
    assert(S.within(std::make_tuple(0, 1), std::make_tuple(0, 1), 1, std::make_tuple(0, 1)).size() == 1);

    auto $ = std::make_tuple(0, -1);

    S.within(0, $, 0, $);

    //assert(S.within({0, 1}, std::make_tuple(0, 1), 1, std::make_tuple(0, 1)).size() == 1);
}




// ============================================================================
int main()
{
    test_selector();

    assert(ndarray<3>(2, 3, 4).size() == 24);
    assert(ndarray<3>(2, 3, 4).shape() == (std::array<int, 3>{2, 3, 4}));

    assert(ndarray<2>(3, 3)[0].shape() == (std::array<int, 1>{3}));
    assert(ndarray<2>(3, 3)[1].shape() == (std::array<int, 1>{3}));
    assert(ndarray<2>(3, 3)[1].strides() == (std::array<int, 1>{1}));

    assert(ndarray<3>(3, 4, 5)[0].shape() == (std::array<int, 2>{4, 5}));
    assert(ndarray<3>(3, 4, 5)[1].shape() == (std::array<int, 2>{4, 5}));
    assert(ndarray<3>(3, 4, 5)[0].strides() == (std::array<int, 2>{5, 1}));

    assert(ndarray<3>(2, 3, 4).offset(0, 0, 0) == 0);
    assert(ndarray<3>(2, 3, 4).offset(0, 0, 1) == 1);
    assert(ndarray<3>(2, 3, 4).offset(0, 1, 0) == 4);
    assert(ndarray<3>(2, 3, 4).offset(0, 2, 1) == 9);
    assert(ndarray<3>(2, 3, 4).offset(1, 2, 1) == 21);

	ndarray<3> A(3, 3, 3);
	ndarray<3> B(3, 3, 3);
    assert(A.shares(A) == true);
    assert(A.shares(B) == false);
    assert(A.shares(A.reshape(9, 3)) == true);

    A(0, 0, 0) = 1;
    A(0, 1, 0) = 2;
    A(0, 1, 2) = 3;

    assert(A(0, 0, 0) == 1);
    assert(A(0, 1, 0) == 2);
    assert(A(0, 1, 2) == 3);
    assert(A.select(1, 1, 3).shape() == (std::array<int, 3>{3, 2, 3}));
    assert(A.select(1, 1, 3)(0, 0, 0) == 2);
    assert(A.select(1, 1, 3)(0, 0, 2) == 3);

    assert(A.within(std::make_tuple(0, 3), std::make_tuple(1, 3), std::make_tuple(2, 3)).shape() == (std::array<int, 3>{3, 2, 1}));
    assert(A.within(std::make_tuple(0, 2), std::make_tuple(1, 3), std::make_tuple(0, 3)).shape() == (std::array<int, 3>{2, 2, 3}));
}
