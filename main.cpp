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


TEST_CASE("selector<2> subset is created properly from a selector<2>", "selector::within")
{
    auto S = selector<2>{{10, 12}, {0, 0}, {10, 12}};
    REQUIRE(S.within(std::make_tuple(0, 10)).reset() == S);
    REQUIRE(S.within(std::make_tuple(2, 4)).reset() == selector<2>{{10, 12}, {2, 0}, {4, 12}});
    REQUIRE(S.within(std::make_tuple(2, 8)).reset().within(std::make_tuple(2, 4)).reset() == selector<2>{{10, 12}, {4, 0}, {6, 12}});
}
