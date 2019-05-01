#include "ndarray.hpp"
#include "catch.hpp"




//=============================================================================
struct test_t
{
	int thing() const &
	{
		return 1;
	}
	int thing() &&
	{
		return 2;
	}
};




//=============================================================================
template<typename... Args>
bool all_equal(Args... args)
{
	auto a = {args...};
	return std::adjacent_find(std::begin(a), std::end(a), std::not_equal_to<>()) == std::end(a);
}

template<typename Function, typename... Args>
bool all_equal_transformed(Function&& fn, Args... args)
{
	auto a = {fn(args)...};
	return std::adjacent_find(std::begin(a), std::end(a), std::not_equal_to<>()) == std::end(a);
}




//=============================================================================
TEST_CASE("rvalue-reference method works as expected")
{
	auto t = test_t();
	REQUIRE(t.thing() == 1);
	REQUIRE(std::move(t).thing() == 2);
}

TEST_CASE("all_equal works correctly")
{
	auto fac = [] (int n, int m) { return nd::make_array(nd::make_unique_provider<double>(n, m)); };
	REQUIRE(all_equal(1));
	REQUIRE(all_equal(1, 1));
	REQUIRE(all_equal(1, 1, 1));
	REQUIRE_FALSE(all_equal(1, 2));
	REQUIRE_FALSE(all_equal(1, 2, 3));
	REQUIRE(all_equal_transformed([] (auto) {return 0;}, 1, 2, 3));
	REQUIRE_FALSE(all_equal_transformed([] (auto i) {return i * 2;}, 1, 2, 3));
	REQUIRE(all_equal_transformed([] (auto c) {return c.size();}, fac(10, 10), fac(10, 10)));
	REQUIRE(all_equal_transformed([] (auto c) {return c.shape();}, fac(10, 10), fac(10, 10)));
	REQUIRE_FALSE(all_equal_transformed([] (auto c) {return c.size();}, fac(10, 10), fac(5, 5)));
	REQUIRE_FALSE(all_equal_transformed([] (auto c) {return c.shape();}, fac(10, 10), fac(5, 5)));
}

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

TEST_CASE("can transform a range", "[transformed_container]")
{
	auto n = 0;

	for (auto a : nd::range(10) | [] (auto a) { return 2 * a; })
	{
		REQUIRE(a == 2 * n);
		++n;
	}
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

TEST_CASE("can create strides", "[memory_strides]")
{
	auto strides = nd::make_strides_row_major(nd::make_shape(20, 10, 5));
	REQUIRE(strides[0] == 50);
	REQUIRE(strides[1] == 5);
	REQUIRE(strides[2] == 1);
	REQUIRE(strides.compute_offset(1, 1, 1) == 56);
}

TEST_CASE("array can be constructed with identity provider", "[array] [identity_provider]")
{
	auto A = nd::make_array(nd::make_index_provider(10), nd::make_access_pattern(10));
	auto B = nd::make_array(nd::make_index_provider(10));
	REQUIRE(A(10) == nd::make_index(10));
	REQUIRE(B(10) == nd::make_index(10));
}

TEST_CASE("uniform provider can be constructed", "[uniform_provider]")
{
	auto p = nd::make_uniform_provider(1.0, 10, 20, 40);
	auto q = p.reshape(nd::make_shape(5, 2, 10, 2, 20, 2));
	REQUIRE(p(nd::make_index(0, 0, 0)) == 1.0);
	REQUIRE(p(nd::make_index(9, 19, 39)) == 1.0);
	REQUIRE(q(nd::make_index(0, 0, 0, 0, 0, 0)) == 1.0);
	REQUIRE(q(nd::make_index(4, 1, 9, 1, 19, 1)) == 1.0);
	REQUIRE(p.size() == q.size());
}

TEST_CASE("shared buffer provider can be constructed", "[array] [shared_provider] [unique_provider]")
{
	auto provider = nd::make_unique_provider<double>(20, 10, 5);
	auto data = provider.data();

	provider(1, 0, 0) = 1;
	provider(0, 2, 0) = 2;
	provider(0, 0, 3) = 3;

	REQUIRE(provider(1, 0, 0) == 1);
	REQUIRE(provider(0, 2, 0) == 2);
	REQUIRE(provider(0, 0, 3) == 3);

	SECTION("can move the provider into a mutable array and get the same data")
	{
		auto A = nd::make_array(std::move(provider));
		A(1, 2, 3) = 123;
		REQUIRE(provider.data() == nullptr);
		REQUIRE(A(1, 0, 0) == 1);
		REQUIRE(A(0, 2, 0) == 2);
		REQUIRE(A(0, 0, 3) == 3);
		REQUIRE(A(1, 2, 3) == 123);
		REQUIRE(A.get_provider().data() == data);

		static_assert(std::is_same<decltype(A)::provider_type, nd::unique_provider_t<3, double>>::value);
	}
	SECTION("can copy a mutable version of the provider into an array and get different data")
	{
		auto A = nd::make_array(provider.shared());
		REQUIRE(provider.data() != nullptr);
		REQUIRE(A(1, 0, 0) == 1);
		REQUIRE(A(0, 2, 0) == 2);
		REQUIRE(A(0, 0, 3) == 3);
		REQUIRE(A.get_provider().data() != data);
	}
	SECTION("can move a mutable version of the provider into an array and get the same data")
	{
		auto A = nd::make_array(std::move(provider).shared());
		REQUIRE(provider.data() == nullptr);
		REQUIRE(A(1, 0, 0) == 1);
		REQUIRE(A(0, 2, 0) == 2);
		REQUIRE(A(0, 0, 3) == 3);
		REQUIRE(A.get_provider().data() == data);
	}
	SECTION("can create a transient array from an immutable one")
	{
		auto A = nd::make_array(std::move(provider).shared());
		auto a = A;
		auto B = A.unique();
		auto b = B;
		auto C = B.shared(); // cannot assign to C
		B(1, 2, 3) = 123;
		REQUIRE(A(1, 2, 3) != 123);
		REQUIRE(B(1, 2, 3) == 123);
		REQUIRE(a.get_provider().data() == A.get_provider().data());
		REQUIRE(b.get_provider().data() != B.get_provider().data());
	}
}

TEST_CASE("shared and unique providers can be built from a function provider", "[evaluate_as_unique] [evaluate_as_shared")
{
	auto A = nd::evaluate_as_unique(nd::make_index_provider(10));
	auto B = nd::evaluate_as_shared(nd::make_index_provider(10));
}

TEST_CASE("zipped provider can be constructed", "[zipped_provider] [zip_arrays]")
{
	auto fac = [] (int n, int m) { return nd::make_array(nd::make_unique_provider<double>(n, m)); };
	auto zipped = nd::zip_arrays(fac(10, 10), fac(10, 10));

	REQUIRE(zipped(nd::make_index(0, 0)) == std::make_tuple(0, 0));

	SECTION("zip_arrays throws if the arrays have diferent shapes")
	{
		REQUIRE_THROWS(nd::zip_arrays(fac(10, 10), fac(9, 9)));
	}
}

TEST_CASE("providers can be reshaped", "[unique_provider] [shared_provider] [reshape]")
{
	SECTION("unique")
	{
		auto provider = nd::make_unique_provider<double>(10, 10);
		REQUIRE_NOTHROW(provider.reshape(nd::make_shape(10, 10)));
		REQUIRE_NOTHROW(provider.reshape(nd::make_shape(5, 20)));
		REQUIRE_NOTHROW(provider.reshape(nd::make_shape(5, 5, 4)));
		REQUIRE_THROWS(provider.reshape(nd::make_shape(10, 10, 10)));
	}
	SECTION("shared")
	{
		auto provider = nd::make_shared_provider<double>(10, 10);
		REQUIRE_NOTHROW(provider.reshape(nd::make_shape(10, 10)));
		REQUIRE_NOTHROW(provider.reshape(nd::make_shape(5, 20)));
		REQUIRE_NOTHROW(provider.reshape(nd::make_shape(5, 5, 4)));
		REQUIRE_THROWS(provider.reshape(nd::make_shape(10, 10, 10)));
		REQUIRE(provider.reshape(nd::make_shape(5, 5, 4)).data() == provider.data());
	}
}

TEST_CASE("arrays can be reshaped given a reshapable provider", "[unique_provider] [reshape]")
{
	using nd::reshape;

	auto A = nd::make_array(nd::make_unique_provider<double>(10, 10));
	REQUIRE_NOTHROW(A|reshape(2, 50));
	REQUIRE_THROWS(A|reshape(2, 51));

	// A.select(nd::make_access_pattern(10, 10, 10).with_start(2, 2, 2).with_jumps(2, 2, 2));
}
