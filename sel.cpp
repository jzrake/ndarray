#include <array>
#include <numeric>
#include <algorithm>
#include <functional>
#include <iostream>
#include "catch.hpp"




// ============================================================================
template<int Rank, int Axis = 0>
class selector
{
public:


    enum { rank = Rank, axis = Axis };


    // ========================================================================
    template<typename... Dims>
    selector(Dims... dims)
    {
        static_assert(sizeof...(Dims) == rank,
            "selector: number of count arguments must match rank");

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
        _final[axis] = start[axis] * count[axis + 1] + final[axis + 1];
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
        return range(start_index, start_index + 1).reset().collapse();
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

        while (index[n] == final[n])
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


private:


    // ========================================================================
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
    std::array<int, rank> count;
    std::array<int, rank> start;
    std::array<int, rank> final;
    std::array<int, rank> skips;


    // ========================================================================
    template<int other_rank, int other_axis>
    friend class selector;
};




// ============================================================================
TEST_CASE("selector<3> does construct and compare correctly ", "[selector]")
{
    auto S = selector<3>(10, 12, 14);
    REQUIRE(S.strides() == std::array<int, 3>{168, 14, 1});
    REQUIRE(S.shape() == std::array<int, 3>{10, 12, 14});
    REQUIRE(S == S.select(std::make_tuple(0, 10, 1)).on<0>());
    REQUIRE(S != S.select(std::make_tuple(0, 10, 2)).on<0>());


    REQUIRE(S.select(0).rank == 2);
    REQUIRE(S.select(0).shape() == std::array<int, 2>{12, 14});
    REQUIRE(S.select(1).shape() == std::array<int, 2>{12, 14});

    REQUIRE(S.select(0, 0).shape() == std::array<int, 1>{14});
    REQUIRE(S.select(1, 0).shape() == std::array<int, 1>{14});
    REQUIRE(S.select(0, 1).shape() == std::array<int, 1>{14});

    REQUIRE(S.select(std::make_tuple(0, 10, 1)).rank == 3);
    REQUIRE(S.select(std::make_tuple(0, 10, 1), std::make_tuple(0, 10, 1)).rank == 3);

    REQUIRE(S.select(0, std::make_tuple(0, 10, 1), std::make_tuple(0, 10, 1)).rank == 2);
    REQUIRE(S.select(std::make_tuple(0, 10, 1), 0, std::make_tuple(0, 10, 1)).rank == 2);
    REQUIRE(S.select(std::make_tuple(0, 10, 1), std::make_tuple(0, 10, 1), 0).rank == 2);

    REQUIRE(S.select(0, std::make_tuple(0, 10, 1), 0).rank == 1);
    REQUIRE(S.select(0, 0, std::make_tuple(0, 10, 1)).rank == 1);
    REQUIRE(S.select(std::make_tuple(0, 10, 1), 0, 0).rank == 1);
}


TEST_CASE("selector<3> does collapse operations correctly", "[selector::collapse]")
{
    auto S = selector<3>(10, 12, 14);
    REQUIRE(S.on<0>().collapse().strides() == std::array<int, 2>{14, 1});
    REQUIRE(S.on<1>().collapse().strides() == std::array<int, 2>{168, 1});
    REQUIRE(S.on<0>().slice(0, 10, 2).strides() == std::array<int, 3>{336, 14, 1});
    REQUIRE(S.on<1>().slice(0, 12, 2).strides() == std::array<int, 3>{168, 28, 1});
    REQUIRE(S.on<2>().slice(0, 14, 2).strides() == std::array<int, 3>{168, 14, 2});
    REQUIRE(S.on<0>().slice(0, 10, 2).shape() == std::array<int, 3>{5, 12, 14});
    REQUIRE(S.on<1>().slice(0, 12, 2).shape() == std::array<int, 3>{10, 6, 14});
    REQUIRE(S.on<2>().slice(0, 14, 2).shape() == std::array<int, 3>{10, 12, 7});

}


TEST_CASE("selector<3> does select operations correctly", "[selector::select]")
{
    auto S = selector<3>(10, 12, 14);
    REQUIRE(S.on<0>().slice(0, 10, 2).strides() == std::array<int, 3>{336, 14, 1});
    REQUIRE(S.on<1>().slice(0, 12, 2).strides() == std::array<int, 3>{168, 28, 1});
    REQUIRE(S.on<2>().slice(0, 14, 2).strides() == std::array<int, 3>{168, 14, 2});
    REQUIRE(S.on<0>().slice(0, 10, 2).shape() == std::array<int, 3>{5, 12, 14});
    REQUIRE(S.on<1>().slice(0, 12, 2).shape() == std::array<int, 3>{10, 6, 14});
    REQUIRE(S.on<2>().slice(0, 14, 2).shape() == std::array<int, 3>{10, 12, 7});
}


TEST_CASE("selector<1> selects correctly with skip applied multiple times", "[selector::select]")
{
    auto S = selector<1>(64);

    REQUIRE(S.skip(2).shape()[0] == 32);
    REQUIRE(S.skip(2).on<0>().skip(2).shape()[0] == 16);
    REQUIRE(S.skip(2).on<0>().skip(2).on<0>().skip(2).shape()[0] == 8);
}


TEST_CASE("selector<4> skips on all dimensions correctly", "[selector::select]")
{
    auto S = selector<4>(2, 4, 6, 8);
    REQUIRE(S.skip(2).skip(4).skip(6).skip(8).size() == 1);
}


TEST_CASE("selector<1> next advances properly", "[selector::next]")
{
    auto S = selector<1>(10);
    auto I = std::array<int, 1>{0};
    auto i = 0;

    do {
        REQUIRE(i == I[0]);
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
        REQUIRE(i == I[0]);
        REQUIRE(j == I[1]);

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
        REQUIRE(index == I);
        S.next(I);
    }
}
