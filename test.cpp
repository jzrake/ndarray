#include "ndarray.hpp"
#include "catch.hpp"




//=============================================================================
TEST_CASE("shapes can be constructed", "[shape]")
{
    auto shape1 = nd::make_shape(10, 10, 10);
    auto shape2 = nd::make_shape(10, 10, 5);
    REQUIRE(shape1 != shape2);
    REQUIRE_FALSE(shape1 == shape2);
    REQUIRE(shape1.size() == 1000);
    REQUIRE(shape1.contains(0, 0, 0));
    REQUIRE(shape1.contains(9, 9, 9));
    REQUIRE_FALSE(shape1.contains(10, 9, 9));
}

TEST_CASE("range can be constructed", "[distance] [enumerate] [range]")
{
    REQUIRE(nd::distance(nd::enumerate(nd::range(10))) == 10);
}

TEST_CASE("buffer can be constructed from", "[buffer]")
{
    SECTION("Can instantiate an empty buffer")
    {
        nd::buffer_t<double> B;
        REQUIRE(B.size() == 0);
        REQUIRE(B.data() == nullptr);
    }

    SECTION("Can instantiate a constant buffer")
    {
        nd::buffer_t<double> B(100, 1.5);
        REQUIRE(B.size() == 100);
        REQUIRE(B.data() != nullptr);
        REQUIRE(B[0] == 1.5);
        REQUIRE(B[99] == 1.5);
    }

    SECTION("Can instantiate a buffer from input iterator")
    {
        std::vector<int> A{0, 1, 2, 3};
        nd::buffer_t<double> B(A.begin(), A.end());
        REQUIRE(B.size() == 4);
        REQUIRE(B[0] == 0);
        REQUIRE(B[1] == 1);
        REQUIRE(B[2] == 2);
        REQUIRE(B[3] == 3);
    }

    SECTION("Can move-construct and move-assign a buffer")
    {
        nd::buffer_t<double> A(100, 1.5);
        nd::buffer_t<double> B(200, 2.0);

        B = std::move(A);

        REQUIRE(A.size() == 0);
        REQUIRE(A.data() == nullptr);

        REQUIRE(B.size() == 100);
        REQUIRE(B[0] == 1.5);
        REQUIRE(B[99] == 1.5);

        auto C = std::move(B);

        REQUIRE(B.size() == 0);
        REQUIRE(B.data() == nullptr);
        REQUIRE(C.size() == 100);
        REQUIRE(C[0] == 1.5);
        REQUIRE(C[99] == 1.5);
    }

    SECTION("Equality operators between buffers work correctly")
    {
        nd::buffer_t<double> A(100, 1.5);   
        nd::buffer_t<double> B(100, 1.5);
        nd::buffer_t<double> C(200, 1.5);
        nd::buffer_t<double> D(100, 2.0);

        REQUIRE(A == A);
        REQUIRE(A == B);
        REQUIRE(A != C);
        REQUIRE(A != D);

        REQUIRE(B == A);
        REQUIRE(B == B);
        REQUIRE(B != C);
        REQUIRE(B != D);

        REQUIRE(C != A);
        REQUIRE(C != B);
        REQUIRE(C == C);
        REQUIRE(C != D);

        REQUIRE(D != A);
        REQUIRE(D != B);
        REQUIRE(D != C);
        REQUIRE(D == D);
    }
}

TEST_CASE("access patterns work OK", "[access_pattern]")
{
	SECTION("can be constructed with factory")
	{
		REQUIRE(nd::make_access_pattern(10, 10, 10).size() == 1000);
		REQUIRE(nd::make_access_pattern(10, 10, 10).with_jumps(2, 2, 2).size() == 125);
	}
	SECTION("can be iterated over")
	{
		auto pat = nd::make_access_pattern(5, 5);
		REQUIRE(nd::distance(pat) == pat.size());
		REQUIRE(pat.contains(0, 0));
		REQUIRE_FALSE(pat.contains(0, 5));
		REQUIRE_FALSE(pat.contains(5, 0));
	}
}

TEST_CASE("array can be constructed with identity provider", "[array] [identity_provider]")
{
	auto A = nd::make_array(nd::make_identity_provider(10), nd::make_access_pattern(10));
	auto B = nd::make_array(nd::make_identity_provider(10));
	REQUIRE(A(10) == nd::make_index(10));
	REQUIRE(B(10) == nd::make_index(10));
}
