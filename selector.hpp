#pragma once
#include <array>
#include <numeric>
#include <functional>
#include "shape.hpp"




// ============================================================================
namespace nd // ND_API_START
{
    template<int Rank, int Axis> struct selector;

    /**
     * Creates a selector without count (memory extent) information, e.g:
     *
     * auto sel = make_selector(_0|100|2, _|50|100);
     *
     * This selector can return size, skips, and shape information, but thinks
     * it has a count of 1 on each axis.
     */
    template<typename... Index>
    static inline auto make_selector(Index... index)
    {
        return selector<sizeof...(Index), 0>().select(index...);
    }

    /**
     * Returns a selector with a different count, by reading from the given
     * iterator range. The iterator difference must match the selector rank,
     * but no checking is performed that the existing selection is within the
     * new count.
     */
    template<typename Selector, typename First, typename Second>
    static inline auto with_count(Selector sel, First begin, Second end)
    {
        auto it = begin;
        auto n = 0;

        if (end - begin != sel.rank)
        {
            throw std::invalid_argument("with_count got wrong number of axes");
        }
        while (it != end)
        {
            sel.count[n++] = int(*it++);
        }
        return sel;
    }
} // ND_API_END




// ============================================================================
template<int Rank, int Axis = 0> // ND_IMPL_START
struct nd::selector
{


    enum { rank = Rank, axis = Axis };


    // ========================================================================
    selector()
    {
        for (int n = 0; n < rank; ++n)
        {
            count[n] = 1;
            start[n] = 0;
            final[n] = 1;
            skips[n] = 1;
        }
    }

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
    template <int R = rank, int A = axis, typename std::enable_if<A == rank - 1>::type* = nullptr>
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
        _final[axis - 1] = count[axis] * final[axis - 1] + start[axis];
        _skips[axis - 1] = count[axis];

        return {_count, _start, _final, _skips};
    }

    template <int R = rank, int A = axis, typename std::enable_if<A < rank - 1>::type* = nullptr>
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
        _final[axis] = count[axis + 1] * final[axis] + start[axis + 1];
        _skips[axis] = 1;

        return {_count, _start, _final, _skips};
    }

    selector<rank, axis + 1> skip(int skips_index) const
    {
        return select(std::make_tuple(start[axis], final[axis], skips_index));
    }

    selector<rank, axis + 1> slice(int lower_index, int upper_index, int skips_index) const
    {
        static_assert(axis < rank, "selector: cannot select on axis >= rank");

        auto _count = count;
        auto _start = start;
        auto _final = final;
        auto _skips = skips;

        _start[axis] = start[axis] + lower_index;
        _final[axis] = start[axis] + upper_index;
        _skips[axis] = skips[axis] * skips_index;

        return {_count, _start, _final, _skips};
    }

    selector<rank, axis + 1> select(axis::selection selection) const
    {
        return slice(selection.lower, selection.upper, selection.skips);
    }

    selector<rank, axis + 1> select(axis::range range) const
    {
        return slice(range.lower, range.upper, 1);
    }

    auto select(axis::index index) const
    {
        return slice(index.lower);
    }

    selector<rank, axis + 1> select(axis::all) const
    {
        return {count, start, final, skips};
    }

    selector<rank, axis + 1> select(std::tuple<int, int, int> selection) const
    {
        return slice(
            std::get<0>(selection),
            std::get<1>(selection),
            std::get<2>(selection));
    }

    selector<rank, axis + 1> select(std::tuple<int, int> range) const
    {
        return slice(std::get<0>(range), std::get<1>(range), 1);
    }

    auto select(int index) const
    {
        return slice(index, index + 1, 1).drop().collapse();
    }

    template<typename First, typename... Rest>
    auto select(First first, Rest... rest) const
    {
        return select(first).select(rest...);
    }

    template<int new_axis>
    selector<rank, new_axis> on() const
    {
        static_assert(new_axis >= 0 && new_axis < rank, "invalid selector axis");
        return {count, start, final, skips};
    }

    selector<rank> reset() const
    {
        return {count, start, final, skips};
    }

    selector<rank, axis - 1> drop() const
    {
        static_assert(axis > 0, "invalid selector axis");
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
        return s;
    }

    std::array<int, rank> shape() const
    {
        std::array<int, rank> s;

        for (int n = 0; n < rank; ++n)
        {
            s[n] = shape(n);
        }
        return s;
    }

    int shape(int axis) const
    {
        return final[axis] / skips[axis] - start[axis] / skips[axis];
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

    std::size_t size() const
    {
        auto s = shape();
        return std::accumulate(s.begin(), s.end(), 1, std::multiplies<int>());
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
    bool contains(Index... index) const
    {
        static_assert(sizeof...(Index) == rank, "selector: index size must match rank");

        auto S = shape::make_shape(index...);

        for (int n = 0; n < rank; ++n)
        {
            auto start_index = std::get<0>(S[n]);
            auto final_index = std::get<1>(S[n]);

            if (start_index < 0 || final_index > final[n] / skips[n] - start[n] / skips[n])
            {
                return false;
            }
        }
        return true;
    }

    selector<rank, axis> shift(int dist) const
    {
        auto sel = *this;
        sel.start[axis] = std::max(sel.start[axis] + dist * skips[axis], 0);
        sel.final[axis] = std::min(sel.final[axis] + dist * skips[axis], sel.count[axis]);
        return sel;
    }




    // ========================================================================
    class iterator
    {
    public:
        iterator() {}
        iterator(selector<rank> sel, std::array<int, rank> ind) : sel(sel), ind(ind) {}
        iterator& operator++() { sel.next(ind); return *this; }
        iterator operator++(int) { auto ret = *this; this->operator++(); return ret; }
        bool operator==(iterator other) const { return ind == other.ind; }
        bool operator!=(iterator other) const { return ind != other.ind; }
        const std::array<int, rank>& operator*() const { return ind; }
    private:
        selector<rank> sel;
        std::array<int, rank> ind;
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
    friend struct selector;
}; // ND_IMPL_END




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


TEST_CASE("make_selector works correctly", "[make_selector]")
{
    auto _ = axis::all();
    CHECK(make_selector(_|0|2).count[0] == 1);
    CHECK(make_selector(_|0|2).start[0] == 0);
    CHECK(make_selector(_|0|2).final[0] == 2);
    CHECK(make_selector(_|0|2).skips[0] == 1);
    CHECK(make_selector(_|0|5) == make_selector(_|0|5|1));
}


TEST_CASE("selector<2> does select-collapse operations correctly", "[selector::select]")
{
    auto S = selector<2>(3, 4);

    SECTION("Selections collapsing axis 0 @ i = 0 have the correct count, stride, and shape")
    {
        CHECK(S.select(0, std::make_tuple(0, 4)).count     == std::array<int, 1>{12});
        CHECK(S.select(0, std::make_tuple(0, 4)).start     == std::array<int, 1>{0});
        CHECK(S.select(0, std::make_tuple(0, 4)).final     == std::array<int, 1>{4});
        CHECK(S.select(0, std::make_tuple(0, 4)).strides() == std::array<int, 1>{1});
        CHECK(S.select(0, std::make_tuple(0, 4)).shape()   == std::array<int, 1>{4});
    }

    SECTION("Selections collapsing axis 0 @ i = 1 have the correct count, stride, and shape")
    {
        CHECK(S.select(1, std::make_tuple(0, 4)).count     == std::array<int, 1>{12});
        CHECK(S.select(1, std::make_tuple(0, 4)).start     == std::array<int, 1>{4});
        CHECK(S.select(1, std::make_tuple(0, 4)).final     == std::array<int, 1>{8});
        CHECK(S.select(1, std::make_tuple(0, 4)).skips     == std::array<int, 1>{1});
        CHECK(S.select(1, std::make_tuple(0, 4)).shape()   == std::array<int, 1>{4});
    }

    SECTION("Selections collapsing a subset of axis 0 @ i = 0 have the correct count, stride, and shape")
    {
        CHECK(S.select(0, std::make_tuple(0, 2)).count     == std::array<int, 1>{12});
        CHECK(S.select(0, std::make_tuple(0, 2)).start     == std::array<int, 1>{0});
        CHECK(S.select(0, std::make_tuple(0, 2)).final     == std::array<int, 1>{2});
        CHECK(S.select(0, std::make_tuple(0, 2)).skips     == std::array<int, 1>{1});
        CHECK(S.select(0, std::make_tuple(0, 2)).shape()   == std::array<int, 1>{2});
    }

    SECTION("Selections collapsing axis 1 at j = 0 have the correct count, stride, and shape")
    {
        CHECK(S.select(std::make_tuple(0, 3), 0).count     == std::array<int, 1>{12});
        CHECK(S.select(std::make_tuple(0, 3), 0).start     == std::array<int, 1>{0});
        CHECK(S.select(std::make_tuple(0, 3), 0).final     == std::array<int, 1>{12});
        CHECK(S.select(std::make_tuple(0, 3), 0).skips     == std::array<int, 1>{4});
        CHECK(S.select(std::make_tuple(0, 3), 0).shape()   == std::array<int, 1>{3});
    }

    SECTION("Selections collapsing axis 1 at j = 1 have the correct count, stride, and shape")
    {
        CHECK(S.select(std::make_tuple(0, 3), 1).count     == std::array<int, 1>{12}); // [1, 5, 9, 13)
        CHECK(S.select(std::make_tuple(0, 3), 1).start     == std::array<int, 1>{1});
        CHECK(S.select(std::make_tuple(0, 3), 1).final     == std::array<int, 1>{13});
        CHECK(S.select(std::make_tuple(0, 3), 1).skips     == std::array<int, 1>{4});
        CHECK(S.select(std::make_tuple(0, 3), 1).shape()   == std::array<int, 1>{3});
    }

    SECTION("Selections collapsing a subset of axis 1 at j = 0 have the correct count, stride, and shape")
    {
        CHECK(S.select(std::make_tuple(0, 2), 0).count     == std::array<int, 1>{12});
        CHECK(S.select(std::make_tuple(0, 2), 0).start     == std::array<int, 1>{0});
        CHECK(S.select(std::make_tuple(0, 2), 0).final     == std::array<int, 1>{8});
        CHECK(S.select(std::make_tuple(0, 2), 0).skips     == std::array<int, 1>{4});
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

TEST_CASE("selector<2> does select-collapse operations correctly (regression)", "[regression]")
{
    SECTION("Selections on last element are correct")
    {
        auto _ = axis::all();
        auto A = nd::selector<2>(5, 5);
        CHECK(A.select(_, 1).size() == 5);
        CHECK(A.select(1, _).size() == 5);

        CHECK(A.select(_, 1).count[0] == 25);
        CHECK(A.select(_, 1).start[0] == 1);
        CHECK(A.select(_, 1).final[0] == 26);
        CHECK(A.select(_, 1).skips[0] == 5);
        CHECK(A.select(_, 1).shape(0) == 5);

        CHECK(A.select(1, _).count[0] == 25);
        CHECK(A.select(1, _).start[0] == 5);
        CHECK(A.select(1, _).final[0] == 10);
        CHECK(A.select(1, _).skips[0] == 1);
        CHECK(A.select(1, _).shape(0) == 5);

        CHECK(A.select(_, 4).count[0] == 25);
        CHECK(A.select(_, 4).start[0] == 4);
        CHECK(A.select(_, 4).final[0] == 29);
        CHECK(A.select(_, 4).skips[0] == 5);
        CHECK(A.select(_, 4).shape(0) == 5);

        CHECK(A.select(4, _).count[0] == 25);
        CHECK(A.select(4, _).start[0] == 20);
        CHECK(A.select(4, _).final[0] == 25);
        CHECK(A.select(4, _).skips[0] == 1);
        CHECK(A.select(4, _).shape(0) == 5);
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
    auto S = selector<2>(10, 10).slice(2, 8, 1).slice(4, 6, 1);    
    auto I = std::array<int, 2>{2, 4};

    for (auto index : S)
    {
        CHECK(index == I);
        S.next(I);
    }
}


TEST_CASE("selector can be shifed", "[selector::shift]")
{
    CHECK(selector<2>(10, 5).on<0>().shift(+2).shape()[0] ==  8);
    CHECK(selector<2>(10, 5).on<0>().shift(+2).shape()[1] ==  5);
    CHECK(selector<2>(10, 5).on<0>().shift(-1).shape()[0] ==  9);
    CHECK(selector<2>(10, 5).on<0>().shift(-1).shape()[1] ==  5);
    CHECK(selector<2>(10, 5).on<1>().shift(-2).shape()[0] == 10);
    CHECK(selector<2>(10, 5).on<1>().shift(-2).shape()[1] ==  3);
    CHECK(selector<2>(10, 5).on<1>().shift(+1).shape()[0] == 10);
    CHECK(selector<2>(10, 5).on<1>().shift(+1).shape()[1] ==  4);
}

#endif // TEST_SELECTOR
