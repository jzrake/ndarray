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
    auto a = {std::forward<Args>(args)...};
    return std::adjacent_find(std::begin(a), std::end(a), std::not_equal_to<>()) == std::end(a);
}

template<typename Function, typename... Args>
bool all_equal_transformed(Function&& fn, Args&&... args)
{
    auto a = {fn(std::forward<Args>(args))...};
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
    REQUIRE(shape1.size() == 3);
    REQUIRE(shape1.volume() == 1000);
    REQUIRE(shape1.contains(0, 0, 0));
    REQUIRE(shape1.contains(9, 9, 9));
    REQUIRE_FALSE(shape1.contains(10, 9, 9));
}

TEST_CASE("shapes support insertion and removal of elements")
{
    auto shape = nd::make_shape(0, 1, 2);
    REQUIRE(shape.remove_elements(nd::make_index(0, 1)) == nd::make_shape(2));
    REQUIRE(shape.remove_elements(nd::make_index(1, 2)) == nd::make_shape(0));
    REQUIRE(shape.remove_elements(nd::make_index(0, 2)) == nd::make_shape(1));
    REQUIRE(shape.insert_elements(nd::make_index(0, 1), nd::make_shape(8, 9)) == nd::make_shape(8, 9, 0, 1, 2));
    REQUIRE(shape.insert_elements(nd::make_index(1, 2), nd::make_shape(8, 9)) == nd::make_shape(0, 8, 9, 1, 2));
    REQUIRE(shape.insert_elements(nd::make_index(2, 3), nd::make_shape(8, 9)) == nd::make_shape(0, 1, 8, 9, 2));
    REQUIRE(shape.insert_elements(nd::make_index(3, 4), nd::make_shape(8, 9)) == nd::make_shape(0, 1, 2, 8, 9));
}

TEST_CASE("can zip, transform, enumerate a range", "[range] [transform] [zip]")
{
    auto n = 0;

    for (auto a : nd::range(10) | [] (auto a) { return 2 * a; })
    {
        REQUIRE(a == 2 * n);
        ++n;
    }
    for (auto&& [m, n] : nd::zip(nd::range(10), nd::range(10)))
    {
        REQUIRE(m == n);
    }
    for (auto&& [m, n] : enumerate(nd::range(10)))
    {
        REQUIRE(m == n);
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
        REQUIRE(nd::distance(pat) == long(pat.size()));
        REQUIRE(pat.contains(0, 0));
        REQUIRE_FALSE(pat.contains(0, 5));
        REQUIRE_FALSE(pat.contains(5, 0));
    }
    SECTION("contains indexes as expected")
    {
        auto pat = nd::make_access_pattern(10).with_start(4).with_jumps(2);
        REQUIRE(pat.contains(0));
        REQUIRE(pat.contains(2));
        REQUIRE_FALSE(pat.contains(3));
    }
    SECTION("generates indexes as expected")
    {
        auto pat = nd::make_access_pattern(10).with_start(4).with_jumps(2);
        REQUIRE(pat.generates(4));
        REQUIRE(pat.generates(6));
        REQUIRE(pat.generates(8));
        REQUIRE_FALSE(pat.generates(0));
        REQUIRE_FALSE(pat.generates(5));
    }
    SECTION("can map and un-map indexes")
    {
        auto pat = nd::make_access_pattern(10).with_start(4).with_jumps(2);
        REQUIRE(pat.inverse_map_index(pat.map_index(nd::make_index(6))) == nd::make_index(6));
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

TEST_CASE("array can be constructed with an index provider", "[array] [index_provider]")
{
    auto A = nd::make_array(nd::make_index_provider(10));
    REQUIRE(A(5) == nd::make_index(5));
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
        // auto b = B; // cannot copy-construct B
        auto C = B.shared(); // cannot assign to C
        B(1, 2, 3) = 123;
        REQUIRE(A(1, 2, 3) != 123);
        REQUIRE(B(1, 2, 3) == 123);
        REQUIRE(a.get_provider().data() == A.get_provider().data());
        REQUIRE(A.get_provider().data() != B.get_provider().data());
    }
}

TEST_CASE("shared and unique providers can be built from a function provider", "[evaluate_as_unique] [evaluate_as_shared")
{
    auto A = nd::evaluate_as_unique(nd::make_index_provider(10));
    auto B = nd::evaluate_as_shared(nd::make_index_provider(10));
}

TEST_CASE("zipped provider can be constructed", "[zipped_provider] [make_zipped_provider]")
{
    auto fac = [] (int n, int m) { return nd::make_array(nd::make_shared_provider<double>(n, m)); };
    auto zipped = nd::make_zipped_provider(fac(10, 10), fac(10, 10));

    REQUIRE(zipped(nd::make_index(0, 0)) == std::make_tuple(0, 0));

    SECTION("make_zipped_provider throws if the arrays have diferent shapes")
    {
        REQUIRE_THROWS(nd::make_zipped_provider(fac(10, 10), fac(9, 9)));
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
    auto A = nd::make_array(nd::make_unique_provider<double>(10, 10));
    REQUIRE_NOTHROW(A | nd::reshape(2, 50));
    REQUIRE_THROWS(A | nd::reshape(2, 51));
}

TEST_CASE("switch provider can be constructed", "[switch_provider]")
{
    auto A1 = nd::make_array(nd::make_uniform_provider(1.0, 10, 10));
    auto A2 = nd::make_array(nd::make_uniform_provider(2.0, 10, 10));
    auto predicate = [] (auto index) { return index[0] < 5; };
    auto provider = nd::make_switch_provider(A1, A2, predicate);
    REQUIRE(provider(nd::make_index(4, 0)) == 1.0);
    REQUIRE(provider(nd::make_index(5, 0)) == 2.0);
    REQUIRE_THROWS(nd::make_switch_provider(
        nd::make_array(nd::make_index_provider(10)),
        nd::make_array(nd::make_index_provider(9)), predicate));
}

TEST_CASE("replace operator works as expected", "[op_replace]")
{
    SECTION("trying to replace a region with an array of the wrong size throws")
    {
        auto A1 = nd::make_array(nd::make_index_provider(10));
        auto A2 = nd::make_array(nd::make_index_provider(5));
        auto patch1 = nd::make_access_pattern(10).with_start(5);
        auto patch2 = nd::make_access_pattern(10).with_start(6);
        REQUIRE_NOTHROW(nd::replace(patch1, A2));
        REQUIRE_THROWS(nd::replace(patch2, A2));
    }
    SECTION("replacing all of an array works")
    {
        auto A1 = nd::make_array(nd::make_uniform_provider(1.0, 10));
        auto A2 = nd::make_array(nd::make_uniform_provider(2.0, 10));
        auto patch = nd::make_access_pattern(10);
        auto A3 = A1 | nd::replace(patch, A2);

        for (auto index : A3.get_accessor())
        {
            REQUIRE(A3(index) == 2.0);
        }
    }
    SECTION("replacing the first half of an array with constant values works")
    {
        auto A1 = nd::make_array(nd::make_uniform_provider(1.0, 10));
        auto A2 = nd::make_array(nd::make_uniform_provider(2.0, 5));
        auto patch = nd::make_access_pattern(5);
        auto A3 = A1 | nd::replace(patch, A2);

        for (auto index : A3.get_accessor())
        {
            REQUIRE(A3(index) == (index[0] < 5 ? 2.0 : 1.0));
        }
    }
    SECTION("replacing the second half of an array with constant values works")
    {
        auto A1 = nd::make_array(nd::make_uniform_provider(1.0, 10));
        auto A2 = nd::make_array(nd::make_uniform_provider(2.0, 5));
        auto patch = nd::make_access_pattern(10).with_start(5);
        auto A3 = A1 | nd::replace(patch, A2);

        for (auto index : A3.get_accessor())
        {
            REQUIRE(A3(index) == (index[0] < 5 ? 1.0 : 2.0));
        }
    }
    SECTION("replacing the second half of an array with linear values works")
    {
        auto A1 = nd::make_array(nd::make_index_provider(10));
        auto A2 = nd::make_array(nd::make_index_provider(5));
        auto patch = nd::make_access_pattern(10).with_start(5);
        auto A3 = A1 | nd::replace(patch, A2);

        for (auto index : A3.get_accessor())
        {
            REQUIRE(A3(index)[0] == (index[0] < 5 ? index[0] : index[0] - 5));
        }
    }
    SECTION("replacing every other value works")
    {
        auto A1 = nd::make_array(nd::make_index_provider(10));
        auto A2 = nd::make_array(nd::make_index_provider(5));
        auto patch = nd::make_access_pattern(10).with_start(0).with_jumps(2);
        auto A3 = A1 | nd::replace(patch, A2);

        for (auto index : A3.get_accessor())
        {
            REQUIRE(A3(index)[0] == (index[0] % 2 == 0 ? index[0] / 2 : index[0]));
        }
    }
}

TEST_CASE("transform operator works as expected", "[op_transform]")
{
    SECTION("with index provider")
    {
        auto A1 = nd::make_array(nd::make_index_provider(10));
        auto A2 = A1 | nd::transform([] (auto i) { return i[0] * 2.0; });

        for (auto index : A2.get_accessor())
        {
            REQUIRE(A2(index) == index[0] * 2.0);
        }
    }
    SECTION("with shared provider")
    {
        auto B1 = nd::shared_array<double>(10);
        auto B2 = B1 | nd::transform([] (auto) { return 2.0; });

        for (auto index : B2.get_accessor())
        {
            REQUIRE(B2(index) == 2.0);
        }
    }
    SECTION("with unique provider")
    {
        auto C1 = nd::unique_array<double>(10);
        auto C2 = C1 | nd::transform([] (auto) { return 2.0; });

        for (auto index : C2.get_accessor())
        {
            REQUIRE(C2(index) == 2.0);
        }
    }
}
