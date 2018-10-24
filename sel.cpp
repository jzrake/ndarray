#include "catch.hpp"
#include <array>
#include <numeric>
#include <algorithm>
#include <functional>
#include <iostream>




// ============================================================================
template<int Rank, int Axis = 0>
struct selector
{
    enum { rank = Rank, axis = Axis };

    selector(
        std::array<int, rank> count,
        std::array<int, rank> start,
        std::array<int, rank> final,
        std::array<int, rank> skips)
    : count(count)
    , start(start)
    , final(final)
    , skips(skips)
    {
    }

    template<typename... Dims>
    selector(Dims... dims)
    {
        static_assert(sizeof...(Dims) == rank, "Wrong number of dimension arguments");
        count = {dims...};

        for (int n = 0; n < rank; ++n)
        {
            start[n] = 0;
            final[n] = count[n];
            skips[n] = 1;
        }
    }

    selector<rank - 1, axis> collapse() const
    {
        static_assert(rank > 0, "selector: cannot collapse zero-rank selector");
        static_assert(axis < rank - 1, "selector: cannot collapse final axis");

        std::array<int, rank - 1> _count;
        std::array<int, rank - 1> _start;
        std::array<int, rank - 1> _final;
        std::array<int, rank - 1> _skips;

        for (int n = 0; n < axis; ++n)
        {
            _count[n] = count[n];
            _start[n] = start[n];
            _final[n] = final[n];
            _skips[n] = skips[n];
        }

        for (int n = axis + 1; n < rank - 1; ++n)
        {
            _count[n] = count[n + 1];
            _start[n] = start[n + 1];
            _final[n] = final[n + 1];
            _skips[n] = skips[n + 1];
        }

        _count[axis] = count[axis] * count[axis + 1];
        _start[axis] = start[axis] * count[axis + 1] + start[axis + 1];
        _final[axis] = final[axis] * count[axis + 1] + final[axis + 1];
        _skips[axis] = 1;

        return {_count, _start, _final, _skips};
    }

    selector<rank, axis + 1> select(int start_index, int final_index, int skips_index) const
    {
        auto _count = count;
        auto _start = start;
        auto _final = final;
        auto _skips = skips;

        _start[axis] = start[axis] + start_index;
        _final[axis] = start[axis] + final_index;
        _skips[axis] = skips[axis] * skips_index;

        return {_count, _start, _final, _skips};
    }

    template<int new_axis>
    selector<rank, new_axis> on() const
    {
        return {count, start, final, skips};
    }

    std::array<int, rank> strides() const
    {
        std::array<int, rank> s;
        s[rank - 1] = 1;

        for (int n = rank - 2; n >= 0; --n)
        {
            s[n] = s[n + 1] * count[n + 1];
        }

        for (int n = 0; n < rank; ++n)
        {
            s[n] *= skips[n];
        }
        return s;
    }

    std::array<int, rank> count;
    std::array<int, rank> start;
    std::array<int, rank> final;
    std::array<int, rank> skips;
};




// ============================================================================
template <class T, std::size_t N>
std::ostream& operator<<(std::ostream& o, const std::array<T, N>& arr)
{
    std::copy(arr.cbegin(), arr.cend(), std::ostream_iterator<T>(o, " "));
    return o;
}




// ============================================================================
TEST_CASE("selector<3> passes basic sanity checks", "[selector]")
{
    auto S = selector<3>(10, 12, 14);
    REQUIRE(S.strides() == std::array<int, 3>{168, 14, 1});
    REQUIRE(S.on<0>().collapse().strides() == std::array<int, 2>{14, 1});
    REQUIRE(S.on<1>().collapse().strides() == std::array<int, 2>{168, 1});
    REQUIRE(S.on<0>().select(0, 10, 2).strides() == std::array<int, 3>{336, 14, 1});
    REQUIRE(S.on<1>().select(0, 12, 2).strides() == std::array<int, 3>{168, 28, 1});
    REQUIRE(S.on<2>().select(0, 14, 2).strides() == std::array<int, 3>{168, 14, 2});
}
