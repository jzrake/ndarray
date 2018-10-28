#pragma once
#include <tuple>
#include <array>




// ============================================================================
namespace nd
{
    namespace shape
    {
        template<unsigned long rank>
        std::array<std::tuple<int, int>, rank> promote(std::array<std::tuple<int, int>, rank> shape);
        std::array<std::tuple<int, int>, 1> promote(std::tuple<int, int> range);
        std::array<std::tuple<int, int>, 1> promote(int start_index);
        template<typename First> auto make_shape(First first);
        template<typename Shape1, typename Shape2> auto make_shape(Shape1 shape1, Shape2 shape2);
        template<typename First, typename... Rest> auto make_shape(First first, Rest... rest);        
    }
}




// ============================================================================
template<unsigned long rank>
std::array<std::tuple<int, int>, rank> nd::shape::promote(std::array<std::tuple<int, int>, rank> shape)
{
    return shape;
}

std::array<std::tuple<int, int>, 1> nd::shape::promote(std::tuple<int, int> range)
{
    return {range};
}

std::array<std::tuple<int, int>, 1> nd::shape::promote(int start_index)
{
    return {std::make_tuple(start_index, start_index + 1)};
}

template<typename First>
auto nd::shape::make_shape(First first)
{
    return promote(first);
}

template<typename Shape1, typename Shape2>
auto nd::shape::make_shape(Shape1 shape1, Shape2 shape2)
{
    auto s1 = promote(shape1);
    auto s2 = promote(shape2);
    auto res = std::array<std::tuple<int, int>, s1.size() + s2.size()>();

    for (int n = 0; n < s1.size(); ++n)
        res[n] = s1[n];

    for (int n = 0; n < s2.size(); ++n)
        res[n + s1.size()] = s2[n];

    return res;
}

template<typename First, typename... Rest>
auto nd::shape::make_shape(First first, Rest... rest)
{
    return make_shape(first, make_shape(rest...));
}




// ============================================================================
#ifdef TEST_SHAPE
#include "catch.hpp"
using namespace nd::shape;


TEST_CASE("make_shape works correctly", "[shape]")
{
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
        auto u = make_shape(0, std::make_tuple(1, 2));
        auto v = make_shape(std::make_tuple(0, 1), 1);

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
        auto t = make_shape(0, std::make_tuple(1, 2), 2);
        auto u = make_shape(std::make_tuple(0, 1), 1, 2);
        auto v = make_shape(0, 1, std::make_tuple(2, 3));

        CHECK(t == u);
        CHECK(u == v);
        CHECK(t.size() == 3);
    }
    SECTION("4D shapes are constructed")
    {
    	CHECK(make_shape(10, 10, 10, 10).size() == 4);
    	CHECK(make_shape(10, 10, 10, std::make_tuple(0, 10)).size() == 4);
    }
}

#endif // TEST_SHAPE
