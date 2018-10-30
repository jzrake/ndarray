#pragma once
#include <array>
#include <numeric>
#include <functional>
#include "shape.hpp"




// ============================================================================
namespace nd
{
    template<int Rank, int Axis> struct selector;
}




// ============================================================================
template<int Rank, int Axis = 0>
struct nd::selector
{


    enum { rank = Rank, axis = Axis };


    // ========================================================================
    selector() {}

    template<typename... Dims>
    selector(Dims... dims) : selector(std::array<int, rank>{dims...})
    {
        static_assert(sizeof...(Dims) == rank,
            "selector: number of count arguments must match rank");
    }

    selector(std::array<int, rank> count) : count(count)
    {
        for (int n = 0; n < rank; ++n)
        {
            start[n] = 0;
            final[n] = count[n];
            skips[n] = 1;
        }
    }

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




    // ========================================================================
    template <int R = rank, int A = axis, typename std::enable_if_t<A == rank - 1>* = nullptr>
    selector<rank - 1, axis - 1> collapse() const
    {
        static_assert(rank > 0, "selector: cannot collapse zero-rank selector");

        std::array<int, rank - 1> _count;
        std::array<int, rank - 1> _start;
        std::array<int, rank - 1> _final;
        std::array<int, rank - 1> _skips;

        for (int n = 0; n < rank - 2; ++n)
        {
            _count[n] = count[n];
            _start[n] = start[n];
            _final[n] = final[n];
            _skips[n] = skips[n];
        }

        _count[axis - 1] = count[axis] * count[axis - 1];
        _start[axis - 1] = count[axis] * start[axis - 1] + start[axis];
        _final[axis - 1] = count[axis] * final[axis - 1] + final[axis];
        _skips[axis - 1] = count[axis];

        return {_count, _start, _final, _skips};
    }

    template <int R = rank, int A = axis, typename std::enable_if_t<A < rank - 1>* = nullptr>
    selector<rank - 1, axis> collapse() const
    {
        static_assert(rank > 0, "selector: cannot collapse zero-rank selector");

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

        _count[axis] = count[axis + 1] * count[axis];
        _start[axis] = count[axis + 1] * start[axis] + start[axis + 1];
        _final[axis] = count[axis + 1] * final[axis]; // + final[axis + 1];
        _skips[axis] = 1;

        return {_count, _start, _final, _skips};
    }

    selector<rank, axis + 1> range(int start_index, int final_index) const
    {
        return select(std::make_tuple(start_index, final_index, 1));
    }

    selector<rank, axis + 1> skip(int skips_index) const
    {
        return select(std::make_tuple(start[axis], final[axis], skips_index));
    }

    selector<rank, axis + 1> slice(int start_index, int final_index, int skips_index) const
    {
        return select(std::make_tuple(start_index, final_index, skips_index));
    }

    selector<rank, axis + 1> select(std::tuple<int, int, int> selection) const
    {
        static_assert(axis < rank, "selector: cannot select on axis >= rank");

        auto _count = count;
        auto _start = start;
        auto _final = final;
        auto _skips = skips;

        _start[axis] = start[axis] + std::get<0>(selection);
        _final[axis] = start[axis] + std::get<1>(selection);
        _skips[axis] = skips[axis] * std::get<2>(selection);

        return {_count, _start, _final, _skips};
    }

    selector<rank, axis + 1> select(std::tuple<int, int> start_final) const
    {
        return range(std::get<0>(start_final), std::get<1>(start_final));
    }

    auto select(int start_index) const
    {
        return range(start_index, start_index + 1).drop().collapse();
    }

    template<typename First, typename... Rest>
    auto select(First first, Rest... rest) const
    {
        return select(first).select(rest...);
    }

    template<int new_axis>
    selector<rank, new_axis> on() const
    {
        return {count, start, final, skips};
    }

    selector<rank> reset() const
    {
        return {count, start, final, skips};
    }

    selector<rank, axis - 1> drop() const
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

    std::array<int, rank> shape() const
    {
        std::array<int, rank> s;

        for (int n = 0; n < rank; ++n)
        {
            s[n] = (final[n] - start[n]) / skips[n];
        }
        return s;
    }

    bool empty() const
    {
        for (int n = 0; n < rank; ++n)
        {
            if (count[n] == 0)
            {
                return true;
            }
        }
        return false;
    }

    bool contiguous() const
    {
        for (int n = 0; n < rank; ++n)
        {
            if (start[n] != 0 || final[n] != count[n] || skips[n] != 1)
            {
                return false;
            }
        }
        return true;
    }

    int size() const
    {
        auto s = shape();
        return std::accumulate(s.begin(), s.end(), 1, std::multiplies<>());
    }

    bool operator==(const selector<rank, axis>& other) const
    {
        return count == other.count &&
        start == other.start &&
        final == other.final &&
        skips == other.skips;
    }

    bool operator!=(const selector<rank, axis>& other) const
    {
        return count != other.count ||
        start != other.start ||
        final != other.final ||
        skips != other.skips;
    }

    bool next(std::array<int, rank>& index) const
    {
        int n = rank - 1;

        index[n] += skips[n];

        while (index[n] >= final[n])
        {
            if (n == 0)
            {
                index = final;
                return false;
            }
            index[n] = start[n];

            --n;

            index[n] += skips[n];
        }
        return true;
    }

    template<typename... Index>
    bool contains(Index... index)
    {
        static_assert(sizeof...(Index) == rank, "selector: index size must match rank");

        auto S = shape::make_shape(index...);

        for (int n = 0; n < rank; ++n)
        {
            auto start_index = std::get<0>(S[n]);
            auto final_index = std::get<1>(S[n]);

            if (start_index < 0 || final_index > (final[n] - start[n]) / skips[n])
            {
                return false;
            }
        }
        return true;
    }




    // ========================================================================
    class iterator
    {
    public:
        iterator(selector<rank> sel, std::array<int, rank> ind) : sel(sel), ind(ind) {}
        iterator& operator++() { sel.next(ind); return *this; }
        iterator operator++(int) { auto ret = *this; this->operator++(); return ret; }
        bool operator==(iterator other) const { return ind == other.ind; }
        bool operator!=(iterator other) const { return ind != other.ind; }
        const std::array<int, rank>& operator*() const { return ind; }
    private:
        std::array<int, rank> ind;
        selector<rank> sel;
    };

    iterator begin() const { return {reset(), start}; }
    iterator end() const { return {reset(), final}; }




    // ========================================================================
    std::array<int, rank> count;
    std::array<int, rank> start;
    std::array<int, rank> final;
    std::array<int, rank> skips;




    // ========================================================================
    template<int other_rank, int other_axis>
    friend class selector;
};




// ============================================================================
#ifdef TEST_SELECTOR
#include "catch.hpp"
using namespace nd;


TEST_CASE("selector<3> does construct and compare correctly", "[selector]")
{
    auto S = selector<3>(10, 12, 14);
    CHECK(S.strides() == std::array<int, 3>{168, 14, 1});
    CHECK(S.shape() == std::array<int, 3>{10, 12, 14});
    CHECK(S == S.select(std::make_tuple(0, 10, 1)).on<0>());
    CHECK(S != S.select(std::make_tuple(0, 10, 2)).on<0>());
}


TEST_CASE("selector<2> does select-collapse operations correctly", "[selector::select]")
{
    auto S = selector<2>(3, 4);

    SECTION("Selections collapsing axis 0 @ i = 0 have the correct count, stride, and shape")
    {
        CHECK(S.select(0, std::make_tuple(0, 4)).count     == std::array<int, 1>{3 * 4});
        CHECK(S.select(0, std::make_tuple(0, 4)).start     == std::array<int, 1>{0});
        CHECK(S.select(0, std::make_tuple(0, 4)).final     == std::array<int, 1>{4});
        CHECK(S.select(0, std::make_tuple(0, 4)).strides() == std::array<int, 1>{1});
        CHECK(S.select(0, std::make_tuple(0, 4)).shape()   == std::array<int, 1>{4});
    }

    SECTION("Selections collapsing axis 0 @ i = 1 have the correct count, stride, and shape")
    {
        CHECK(S.select(1, std::make_tuple(0, 4)).count     == std::array<int, 1>{3 * 4});
        CHECK(S.select(1, std::make_tuple(0, 4)).start     == std::array<int, 1>{4});
        CHECK(S.select(1, std::make_tuple(0, 4)).final     == std::array<int, 1>{8});
        CHECK(S.select(1, std::make_tuple(0, 4)).strides() == std::array<int, 1>{1});
        CHECK(S.select(1, std::make_tuple(0, 4)).shape()   == std::array<int, 1>{4});
    }

    SECTION("Selections collapsing a subset of axis 0 @ i = 0 have the correct count, stride, and shape")
    {
        CHECK(S.select(0, std::make_tuple(0, 2)).count     == std::array<int, 1>{3 * 4});
        CHECK(S.select(0, std::make_tuple(0, 2)).start     == std::array<int, 1>{0});
        CHECK(S.select(0, std::make_tuple(0, 2)).final     == std::array<int, 1>{2});
        CHECK(S.select(0, std::make_tuple(0, 2)).strides() == std::array<int, 1>{1});
        CHECK(S.select(0, std::make_tuple(0, 2)).shape()   == std::array<int, 1>{2});
    }

    SECTION("Selections collapsing axis 1 at j = 0 have the correct count, stride, and shape")
    {
        CHECK(S.select(std::make_tuple(0, 3), 0).count     == std::array<int, 1>{3 * 4});
        CHECK(S.select(std::make_tuple(0, 3), 0).start     == std::array<int, 1>{0});
        CHECK(S.select(std::make_tuple(0, 3), 0).final     == std::array<int, 1>{3 * 4 + 1});
        CHECK(S.select(std::make_tuple(0, 3), 0).strides() == std::array<int, 1>{4});
        CHECK(S.select(std::make_tuple(0, 3), 0).shape()   == std::array<int, 1>{3});
    }

    SECTION("Selections collapsing axis 1 at j = 1 have the correct count, stride, and shape")
    {
        CHECK(S.select(std::make_tuple(0, 3), 1).count     == std::array<int, 1>{3 * 4});
        CHECK(S.select(std::make_tuple(0, 3), 1).start     == std::array<int, 1>{1});
        CHECK(S.select(std::make_tuple(0, 3), 1).final     == std::array<int, 1>{3 * 4 + 2});
        CHECK(S.select(std::make_tuple(0, 3), 1).strides() == std::array<int, 1>{4});
        CHECK(S.select(std::make_tuple(0, 3), 1).shape()   == std::array<int, 1>{3});
    }

    SECTION("Selections collapsing a subset of axis 1 at j = 0 have the correct count, stride, and shape")
    {
        CHECK(S.select(std::make_tuple(0, 2), 0).count     == std::array<int, 1>{3 * 4});
        CHECK(S.select(std::make_tuple(0, 2), 0).start     == std::array<int, 1>{0});
        CHECK(S.select(std::make_tuple(0, 2), 0).final     == std::array<int, 1>{2 * 4 + 1});
        CHECK(S.select(std::make_tuple(0, 2), 0).strides() == std::array<int, 1>{4});
        CHECK(S.select(std::make_tuple(0, 2), 0).shape()   == std::array<int, 1>{2});
    }

    SECTION("Selections collapsing only axis 0 have the correct count, stride, and shape")
    {
        CHECK(S.select(0).count == std::array<int, 1>{3 * 4});
        CHECK(S.select(0).start == std::array<int, 1>{0});
        CHECK(S.select(0).final == std::array<int, 1>{4});
        CHECK(S.select(0).strides() == std::array<int, 1>{1});
        CHECK(S.select(0).shape()   == std::array<int, 1>{4});
    }
}


TEST_CASE("selector<1> selects correctly with skip applied multiple times", "[selector::skip]")
{
    auto S = selector<1>(64);

    CHECK(S.skip(2).shape()[0] == 32);
    CHECK(S.skip(2).on<0>().skip(2).shape()[0] == 16);
    CHECK(S.skip(2).on<0>().skip(2).on<0>().skip(2).shape()[0] == 8);
}


TEST_CASE("selector<4> skips on all dimensions correctly", "[selector::skip]")
{
    auto S = selector<4>(2, 4, 6, 8);
    CHECK(S.skip(2).skip(4).skip(6).skip(8).size() == 1);
}


TEST_CASE("selector<1> next advances properly", "[selector::next]")
{
    auto S = selector<1>(10);
    auto I = std::array<int, 1>{0};
    auto i = 0;

    do {
        CHECK(i == I[0]);
        ++i;
    } while (S.next(I));
}


TEST_CASE("selector<2> next advances properly", "[selector::next]")
{
    auto S = selector<2>(10, 10);
    auto I = std::array<int, 2>{0, 0};
    auto i = 0;
    auto j = 0;

    do {
        CHECK(i == I[0]);
        CHECK(j == I[1]);

        if (++j == 10)
        {
            j = 0;
            ++i;
        }
    } while (S.next(I));
}


TEST_CASE("selector<2> subset iterator passes sanity checks", "[selector::iterator]")
{
    auto S = selector<2>(10, 10).range(2, 8).range(4, 6);    
    auto I = std::array<int, 2>{2, 4};

    for (auto index : S)
    {
        CHECK(index == I);
        S.next(I);
    }
}


#endif // TEST_SELECTOR
