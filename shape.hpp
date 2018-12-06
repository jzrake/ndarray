#pragma once
#include <tuple>
#include <array>




// ============================================================================
namespace nd // ND_API_START
{
    namespace axis
    {
        struct selection
        {
            selection() {}
            selection(int lower, int upper, int skips) : lower(lower), upper(upper), skips(skips) {}
            int lower = 0, upper = 0, skips = 1;
        };

        struct range
        {
            range() {}
            range(int lower, int upper) : lower(lower), upper(upper) {}
            selection operator|(int skips) const { return selection(lower, upper, skips); }
            int lower = 0, upper = 0;
        };

        struct index
        {
            index() {}
            index(int lower) : lower(lower) {}
            range operator|(int upper) const { return range(lower, upper); }
            int lower = 0;
        };

        struct all
        {
            index operator|(int lower) const { return index(lower); }
        };
    }

    namespace shape
    {
        template<unsigned long rank>
        static inline std::array<std::tuple<int, int>, rank> promote(std::array<std::tuple<int, int>, rank> shape);
        static inline std::array<std::tuple<int, int>, 1> promote(std::tuple<int, int, int> selection);
        static inline std::array<std::tuple<int, int>, 1> promote(std::tuple<int, int> range);
        static inline std::array<std::tuple<int, int>, 1> promote(int index);
        static inline std::array<std::tuple<int, int>, 1> promote(axis::selection selection);
        static inline std::array<std::tuple<int, int>, 1> promote(axis::range range);
        static inline std::array<std::tuple<int, int>, 1> promote(axis::index index);
        static inline std::array<std::tuple<int, int>, 1> promote(axis::all all);
        template<typename First>                   static inline auto make_shape(First first);
        template<typename First, typename Second>  static inline auto make_shape(First first, Second second);
        template<typename First, typename... Rest> static inline auto make_shape(First first, Rest... rest);
    }
} // ND_API_END




// ============================================================================
template<unsigned long rank> // ND_IMPL_START
std::array<std::tuple<int, int>, rank> nd::shape::promote(std::array<std::tuple<int, int>, rank> shape)
{
    return shape;
}

// std::array<std::tuple<int, int>, 1> nd::shape::promote(std::tuple<int, int, int> selection)
// {
//     return {std::make_tuple(std::get<0>(selection), std::get<1>(selection))};
// }

std::array<std::tuple<int, int>, 1> nd::shape::promote(std::tuple<int, int> range)
{
    return {range};
}

std::array<std::tuple<int, int>, 1> nd::shape::promote(int start_index)
{
    return {std::make_tuple(start_index, start_index + 1)};
}

std::array<std::tuple<int, int>, 1> nd::shape::promote(axis::selection selection)
{
    return {std::make_tuple(selection.lower, selection.upper)};
}

std::array<std::tuple<int, int>, 1> nd::shape::promote(axis::range range)
{
    return {std::make_tuple(range.lower, range.upper)};
}

std::array<std::tuple<int, int>, 1> nd::shape::promote(axis::index index)
{
    return {std::make_tuple(index.lower, index.lower + 1)};
}

std::array<std::tuple<int, int>, 1> nd::shape::promote(axis::all)
{
    return {std::make_tuple(0, -1)};
}

template<typename First>
auto nd::shape::make_shape(First first)
{
    return promote(first);
}

template<typename First, typename Second>
auto nd::shape::make_shape(First first, Second second)
{
    auto s1 = promote(first);
    auto s2 = promote(second);
    auto res = std::array<std::tuple<int, int>, s1.size() + s2.size()>();

    for (std::size_t n = 0; n < s1.size(); ++n)
        res[n] = s1[n];

    for (std::size_t n = 0; n < s2.size(); ++n)
        res[n + s1.size()] = s2[n];

    return res;
}

template<typename First, typename... Rest>
auto nd::shape::make_shape(First first, Rest... rest)
{
    return make_shape(first, make_shape(rest...));
} // ND_IMPL_END




// ============================================================================
#ifdef TEST_SHAPE
#include "catch.hpp"
using namespace nd::shape;


TEST_CASE("make_shape works correctly", "[shape]")
{
    auto _ = nd::axis::all();

    SECTION("1D shapes are constructed")
    {
        auto t = make_shape(0);
        auto u = make_shape(std::make_tuple(0, 10));
        auto v = make_shape(std::array<std::tuple<int, int>, 1>{std::make_tuple(0, 10)});
        static_assert(std::is_same<decltype(t), std::array<std::tuple<int, int>, 1>>::value, "Not OK");
        static_assert(std::is_same<decltype(u), std::array<std::tuple<int, int>, 1>>::value, "Not OK");
        static_assert(std::is_same<decltype(v), std::array<std::tuple<int, int>, 1>>::value, "Not OK");
    }

    SECTION("2D shapes are constructed")
    {
        auto t = make_shape(0, 1);
        auto u = make_shape(0, _|1|2);
        auto v = make_shape(_|0|1, 1);

        CHECK(t == u);
        CHECK(u == v);

        CHECK(t.size() == 2);
        CHECK(std::get<0>(t[0]) == 0);
        CHECK(std::get<1>(t[0]) == 1);
        CHECK(std::get<0>(t[1]) == 1);
        CHECK(std::get<1>(t[1]) == 2);
    }

    SECTION("3D shapes are constructed")
    {
        auto t = make_shape(0, _|1|2, 2);
        auto u = make_shape(_|0|1, 1, 2);
        auto v = make_shape(0, 1, _|2|3);

        CHECK(t == u);
        CHECK(u == v);
        CHECK(t.size() == 3);
    }

    SECTION("4D shapes are constructed")
    {
        CHECK(make_shape(10, 10, 10, 10).size() == 4);
        CHECK(make_shape(10, 10, 10, _|0|10).size() == 4);
    }
}

#endif // TEST_SHAPE
