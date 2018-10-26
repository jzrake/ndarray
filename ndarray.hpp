#include <array>
#include <memory>
#include <vector>
#include <numeric>
#include <cassert>
#include "selector.hpp"




// ============================================================================
#ifndef NDARRAY_NO_EXCEPTIONS
#define NDARRAY_ASSERT_VALID_ARGUMENT(condition, message) do { if (! (condition)) throw std::invalid_argument(message); } while(0)
#else
#define NDARRAY_ASSERT_VALID_ARGUMENT(condition, message) do { if (! (condition)) std::terminate(); } while(0)
#endif




// ============================================================================
namespace nd
{
    template<typename T, int Rank, typename Op> class binary_op;
    template<typename T, int Rank> class ndarray;
    template<typename T> struct dtype_str;
}




// ============================================================================
template<typename T, int Rank, typename Op>
struct nd::binary_op
{
    static ndarray<T, Rank> perform(const ndarray<T, Rank>& A, const ndarray<T, Rank>& B)
    {
        assert(A.shape() == B.shape());

        auto op = Op();
        auto C = ndarray<T, Rank>(A.shape());
        auto a = A.begin();
        auto b = B.begin();
        auto c = C.begin();

        for (; a != A.end(); ++a, ++b, ++c)
            *c = op(*a, *b);

        return C;
    }
    static void perform(ndarray<T, Rank>& A, const ndarray<T, Rank>& B)
    {
        assert(A.shape() == B.shape());

        auto op = Op();
        auto a = A.begin();
        auto b = B.begin();

        for (; a != A.end(); ++a, ++b)
            *a = op(*a, *b);
    }
};




// ============================================================================
template<> struct nd::dtype_str<float > { static std::array<char, 8> value; };
template<> struct nd::dtype_str<double> { static std::array<char, 8> value; };
template<> struct nd::dtype_str<int   > { static std::array<char, 8> value; };
template<> struct nd::dtype_str<long  > { static std::array<char, 8> value; };

std::array<char, 8> nd::dtype_str<float >::value = {'f','4',  0,  0,  0,  0,  0,  0};
std::array<char, 8> nd::dtype_str<double>::value = {'f','8',  0,  0,  0,  0,  0,  0};
std::array<char, 8> nd::dtype_str<int   >::value = {'i','4',  0,  0,  0,  0,  0,  0};
std::array<char, 8> nd::dtype_str<long  >::value = {'i','8',  0,  0,  0,  0,  0,  0};





// ============================================================================
template<typename T, int Rank>
class nd::ndarray
{
public:


    using data_type = T;
    enum { rank = Rank };


    /**
     * Constructors
     * 
     */
    // ========================================================================
    template<int R = rank, typename = typename std::enable_if<R == 0>::type>
    ndarray(T value=T())
    : scalar_offset(0)
    , data(std::make_shared<std::vector<T>>(1, value))
    {
    }

    template<int R = rank, typename = typename std::enable_if<R == 0>::type>
    ndarray(int scalar_offset, std::shared_ptr<std::vector<T>>& data)
    : scalar_offset(scalar_offset)
    , data(data)
    {
    }

    template<int R = rank, typename = typename std::enable_if<R == 1>::type>
    ndarray(std::initializer_list<T> elements)
    : count({int(elements.size())})
    , start({0})
    , final({int(elements.size())})
    , skips({1})
    , strides({1})
    , data(std::make_shared<std::vector<T>>(elements.begin(), elements.end()))
    {
    }

    template<typename... Dims>
    ndarray(Dims... dims) : ndarray(std::array<int, rank>({dims...}))
    {
        static_assert(sizeof...(dims) == rank,
          "Number of arguments to ndarray constructor must match rank");
    }

    template<typename SelectorType>
    ndarray(SelectorType selector, std::shared_ptr<std::vector<T>>& data)
    : count(selector.count)
    , start(selector.start)
    , final(selector.final)
    , skips(selector.skips)
    , strides(compute_strides(count))
    , data(data)
    {
    }

    ndarray() : ndarray(constant_array<rank>(0))
    {
    }

    ndarray(std::array<int, rank> dim_sizes)
    : count(dim_sizes)
    , start(constant_array<rank>(0))
    , final(dim_sizes)
    , skips(constant_array<rank>(1))
    , strides(compute_strides(count))
    , data(std::make_shared<std::vector<T>>(product(dim_sizes)))
    {
    }

    ndarray(std::array<int, rank> dim_sizes, std::shared_ptr<std::vector<T>>& data)
    : count(dim_sizes)
    , start(constant_array<rank>(0))
    , final(dim_sizes)
    , skips(constant_array<rank>(1))
    , strides(compute_strides(count))
    , data(data)
    {
        NDARRAY_ASSERT_VALID_ARGUMENT(
            data->size() == std::accumulate(count.begin(), count.end(), 1, std::multiplies<>()),
            "Size of data buffer is not the product of dim sizes");
    }

    ndarray(
        std::array<int, rank> count,
        std::array<int, rank> start,
        std::array<int, rank> final,
        std::shared_ptr<std::vector<T>>& data)
    : count(count)
    , start(start)
    , final(final)
    , skips(constant_array<rank>(1))
    , strides(compute_strides(count))
    , data(data)
    {
        NDARRAY_ASSERT_VALID_ARGUMENT(
            data->size() == std::accumulate(count.begin(), count.end(), 1, std::multiplies<>()),
            "Size of data buffer is not the product of dim sizes");
    }

    ndarray(const ndarray<T, rank>& other)
    : count(other.shape())
    , start(constant_array<rank>(0))
    , final(other.shape())
    , skips(constant_array<rank>(1))
    , strides(compute_strides(count))
    , data(std::make_shared<std::vector<T>>(size()))
    {
        *this = other;
    }

    ndarray(ndarray<T, rank>& other)
    {
        become(other);
    }




    /**
     * Assignment operators
     * 
     */
    // ========================================================================
    ndarray<T, rank>& operator=(T value)
    {
        for (auto& a : *this)
        {
            a = value;
        }
        return *this;
    }

    ndarray<T, rank>& operator=(const ndarray<T, rank>& other)
    {
        NDARRAY_ASSERT_VALID_ARGUMENT(shape() == other.shape(),
            "assignment from ndarray of incompatible shape");

        auto a = this->begin();
        auto b = other.begin();

        for (; a != end(); ++a, ++b)
        {
            *a = *b;
        }
        return *this;
    }

    void become(ndarray<T, rank>& other)
    {
        count = other.count;
        start = other.start;
        final = other.final;
        skips = other.skips;
        strides = other.strides;
        data = other.data;
    }

    // template<typename... Sizes>
    // void resize(Sizes... sizes)
    // {
    //     become(ndarray<T, rank>(sizes...));
    // }




    /**
     * Factories
     * 
     */
    // ========================================================================
    static ndarray<T, rank> stack(std::initializer_list<ndarray<T, rank - 1>> arrays)
    {
        if (arrays.size() == 0)
        {
            return ndarray<T, rank>();            
        }

        auto required_shape = arrays.begin()->shape();

        std::array<int, rank> dim_sizes;
        dim_sizes[0] = arrays.size();

        for (int n = 1; n < rank; ++n)
        {
            dim_sizes[n] = required_shape[n - 1];
        }
        auto A = ndarray<T, rank>(dim_sizes);
        int n = 0;

        for (const auto& array : arrays)
        {
            A[n] = array;
            ++n;
        }
        return A;
    }

    template<typename... Sizes>
    static ndarray<T, rank> ones(Sizes... sizes)
    {
        auto A = ndarray<T, rank>(sizes...);
        auto x = T();
        for (auto& a : A) a = 1;
        return A;
    }

    template<typename... Sizes>
    static ndarray<T, rank> zeros(Sizes... sizes)
    {
        auto A = ndarray<T, rank>(sizes...);
        auto x = T();
        for (auto& a : A) a = 0;
        return A;
    }

    template<typename... Sizes>
    static ndarray<T, rank> arange(Sizes... sizes)
    {
        auto A = ndarray<T, rank>(sizes...);
        auto x = T();
        for (auto& a : A) a = x++;
        return A;
    }




    /**
     * Shape and size query methods
     * 
     */
    // ========================================================================
    int size() const
    {
        return make_selector().size();
    }

    std::array<int, rank> shape() const
    {
        return make_selector().shape();
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




    /**
     * Data accessors and selection methods
     * 
     */
    // ========================================================================
    template <int R = rank, typename std::enable_if_t<R == 1>* = nullptr>
    ndarray<T, rank - 1> operator[](int index)
    {
        return {offset_relative({index}), data};
    }

    template <int R = rank, typename std::enable_if_t<R == 1>* = nullptr>
    ndarray<T, rank - 1> operator[](int index) const
    {
        auto d = std::make_shared<std::vector<T>>(1, data->operator[](offset_relative({index})));
        return {0, d};
    }

    template <int R = rank, typename std::enable_if_t<R != 1>* = nullptr>
    ndarray<T, rank - 1> operator[](int index)
    {
        return {make_selector().select(index), data};
    }

    template <int R = rank, typename std::enable_if_t<R != 1>* = nullptr>
    ndarray<T, rank - 1> operator[](int index) const
    {
        auto S = make_selector().collapse(index);
        auto d = std::make_shared<std::vector<T>>(S.size());
        auto a = d->begin();
        auto b = begin();

        for ( ; a != d->end(); ++a, ++b)
        {
            *a = *b;
        }
        return {S, d};
    }

    template<typename... Index>
    T& operator()(Index... index)
    {
        static_assert(sizeof...(index) == rank,
          "Number of arguments to ndarray::operator() must match rank");

        return data->operator[](offset_relative({index...}));
    }

    template<typename... Index>
    const T& operator()(Index... index) const
    {
        static_assert(sizeof...(index) == rank,
          "Number of arguments to ndarray::operator() must match rank");

        return data->operator[](offset_relative({index...}));
    }

    template<typename... Index>
    auto select(Index... index)
    {
        auto S = make_selector().select(index...);
        return ndarray<T, S.rank>(S, data);
    }

    operator T() const
    {
        static_assert(rank == 0, "can only convert rank-0 array to scalar value");
        return data->operator[](scalar_offset);
    }

    bool is(const ndarray<T, rank>& other) const
    {
        return (scalar_offset == other.scalar_offset
        && count == other.count
        && start == other.start
        && final == other.final
        && skips == other.skips
        && strides == other.strides
        && data == other.data);
    }

    ndarray<T, rank> copy() const
    {
        auto d = std::make_shared<std::vector<T>>(begin(), end());
        return {shape(), d};
    }

    const std::vector<T>& container() const
    {
        return *data;
    }

    template<int other_rank>
    bool shares(const ndarray<T, other_rank>& other) const
    {
        return data == other.data;
    }




    // ========================================================================
    class iterator
    {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = T*;
        using reference = T&;
        using iterator_category = std::forward_iterator_tag;

        iterator(ndarray<T, rank>& array, typename selector<rank>::iterator it) : array(array), it(it) {}
        iterator& operator++() { it.operator++(); return *this; }
        iterator operator++(int) { auto ret = *this; this->operator++(); return ret; }
        bool operator==(iterator other) const { return array.is(other.array) && it == other.it; }
        bool operator!=(iterator other) const { return array.is(other.array) && it != other.it; }
        T& operator*() { return array.data->operator[](array.offset_absolute(*it)); }
    private:
        typename selector<rank>::iterator it;
        ndarray<T, rank>& array;
    };

    iterator begin() { return {*this, make_selector().begin()}; }
    iterator end() { return {*this, make_selector().end()}; }




    // ========================================================================
    class const_iterator
    {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = const T;
        using pointer = const T*;
        using reference = const T&;
        using iterator_category = std::forward_iterator_tag;

        const_iterator(const ndarray<T, rank>& array, typename selector<rank>::iterator it) : array(array), it(it) {}
        const_iterator& operator++() { it.operator++(); return *this; }
        const_iterator operator++(int) { auto ret = *this; this->operator++(); return ret; }
        bool operator==(const_iterator other) const { return array.is(other.array) && it == other.it; }
        bool operator!=(const_iterator other) const { return array.is(other.array) && it != other.it; }
        const T& operator*() const { return array.data->operator[](array.offset_absolute(*it)); }
    private:
        typename selector<rank>::iterator it;
        const ndarray<T, rank>& array;
    };

    const_iterator begin() const { return {*this, make_selector().begin()}; }
    const_iterator end() const { return {*this, make_selector().end()}; }




    /**
     * Arithmetic operations
     * 
     */
    // ========================================================================
    ndarray<T, rank>& operator+=(T value) { for (auto& a : *this) { a += value; } return *this; }
    ndarray<T, rank>& operator-=(T value) { for (auto& a : *this) { a -= value; } return *this; }
    ndarray<T, rank>& operator*=(T value) { for (auto& a : *this) { a *= value; } return *this; }
    ndarray<T, rank>& operator/=(T value) { for (auto& a : *this) { a /= value; } return *this; }

    ndarray<T, rank>& operator+=(const ndarray<T, rank>& other) { binary_op<T, rank, std::plus      <T>>::perform(*this, other); return *this; }
    ndarray<T, rank>& operator-=(const ndarray<T, rank>& other) { binary_op<T, rank, std::minus     <T>>::perform(*this, other); return *this; }
    ndarray<T, rank>& operator*=(const ndarray<T, rank>& other) { binary_op<T, rank, std::multiplies<T>>::perform(*this, other); return *this; }
    ndarray<T, rank>& operator/=(const ndarray<T, rank>& other) { binary_op<T, rank, std::divides   <T>>::perform(*this, other); return *this; }

    ndarray<T, rank> operator+(T value) const { auto A = copy(); for (auto& a : A) { a += value; } return A; }
    ndarray<T, rank> operator-(T value) const { auto A = copy(); for (auto& a : A) { a -= value; } return A; }
    ndarray<T, rank> operator*(T value) const { auto A = copy(); for (auto& a : A) { a *= value; } return A; }
    ndarray<T, rank> operator/(T value) const { auto A = copy(); for (auto& a : A) { a /= value; } return A; }

    ndarray<T, rank> operator+(const ndarray<T, rank>& other) const { return binary_op<T, rank, std::plus      <T>>::perform(*this, other); }
    ndarray<T, rank> operator-(const ndarray<T, rank>& other) const { return binary_op<T, rank, std::minus     <T>>::perform(*this, other); }
    ndarray<T, rank> operator*(const ndarray<T, rank>& other) const { return binary_op<T, rank, std::multiplies<T>>::perform(*this, other); }
    ndarray<T, rank> operator/(const ndarray<T, rank>& other) const { return binary_op<T, rank, std::divides   <T>>::perform(*this, other); }




    /**
     * Basic serialization operations
     * 
     */
    // ========================================================================
    std::string dumps() const
    {
        // rank  ... 1 int
        // shape ... rank int's
        // data  ... size T's

        auto D = dtype_str<T>::value;
        auto R = int(rank);
        auto S = shape();
        auto str = std::string();

        str.insert(str.end(), (char*)&D, (char*)(&D + 1));
        str.insert(str.end(), (char*)&R, (char*)(&R + 1));
        str.insert(str.end(), (char*)&S, (char*)(&S + 1));

        for (const auto& x : *this)
        {
            str.insert(str.end(), (char*)&x, (char*)(&x + 1));
        }
        return str;
    }

    static ndarray<T, rank> loads(const std::string& str)
    {
        auto it = str.begin();
        auto data = std::make_shared<std::vector<T>>();
        auto D = std::array<char, 8>();
        auto R = int();
        auto S = constant_array<rank>(0);
        auto x = T();

        NDARRAY_ASSERT_VALID_ARGUMENT(it + sizeof(D) <= str.end(), "unexpected end of ndarray header string");
        std::memcpy(&D, &*it, sizeof(D));
        it += sizeof(D);

        NDARRAY_ASSERT_VALID_ARGUMENT(it + sizeof(R) <= str.end(), "unexpected end of ndarray header string");
        std::memcpy(&R, &*it, sizeof(R));
        it += sizeof(R);

        NDARRAY_ASSERT_VALID_ARGUMENT(it + sizeof(S) <= str.end(), "unexpected end of ndarray header string");
        std::memcpy(&S, &*it, sizeof(S));
        it += sizeof(S);

        NDARRAY_ASSERT_VALID_ARGUMENT(D == dtype_str<T>::value, "ndarray string has wrong data type");
        NDARRAY_ASSERT_VALID_ARGUMENT(R == rank, "ndarray string has the wrong rank");

        while (it != str.end())
        {
            NDARRAY_ASSERT_VALID_ARGUMENT(it + sizeof(T) <= str.end(), "unexpected end of ndarray data string");
            std::memcpy(&x, &*it, sizeof(T));
            data->push_back(x);
            it += sizeof(T);
        }

        return {S, data};
    }


private:
    /**
     * Private utility methods
     * 
     */
    // ========================================================================
    int offset_relative(std::array<int, rank> index) const
    {
        int m = scalar_offset;

        for (int n = 0; n < rank; ++n)
        {
            m += start[n] + skips[n] * index[n] * strides[n];
        }
        return m;
    }

    int offset_absolute(std::array<int, rank> index) const
    {
        int m = scalar_offset;

        for (int n = 0; n < rank; ++n)
        {
            m += index[n] * strides[n];
        }
        return m;
    }

    selector<rank> make_selector() const
    {
        return selector<rank>{count, start, final, skips};
    }

    template <int R = rank, typename std::enable_if_t<R == 0>* = nullptr>
    static std::array<int, rank> compute_strides(std::array<int, rank> count)
    {
        return std::array<int, rank>();
    }

    template <int R = rank, typename std::enable_if_t<R != 0>* = nullptr>
    static std::array<int, rank> compute_strides(std::array<int, rank> count)
    {
        return selector<rank>(
            count,
            constant_array<rank>(0),
            count,
            constant_array<rank>(1)).strides();
    }

    template<int length>
    static std::array<int, length> constant_array(T value)
    {
        std::array<int, length> A;
        for (auto& a : A) a = value;
        return A;
    }

    template<typename C>
    static size_t product(const C& c)
    {
        return std::accumulate(c.begin(), c.end(), 1, std::multiplies<>());
    }




    /**
     * Data members
     *
     */
    // ========================================================================
    int scalar_offset = 0;
    std::array<int, rank> count;
    std::array<int, rank> start;
    std::array<int, rank> final;
    std::array<int, rank> skips;
    std::array<int, rank> strides;
    std::shared_ptr<std::vector<T>> data;




    /**
     * Grant friendship to ndarray's of other ranks.
     *
     */
    template<typename, int>
    friend class ndarray;
    friend class iterator;
};




// ============================================================================
#ifdef TEST_NDARRAY
#include "catch.hpp"
using T = double;


TEST_CASE("ndarray can be constructed ", "[ndarray]")
{
    SECTION("trivial construction works OK")
    {
        REQUIRE(ndarray<T, 1>(1).size() == 1);
        REQUIRE(ndarray<T, 1>(1).shape() == std::array<int, 1>{1});
        REQUIRE(ndarray<T, 1>().empty());
        REQUIRE_FALSE(ndarray<T, 1>(1).empty());
    }

    SECTION("ndarray constructor throws if the data buffer has the wrong size")
    {
        auto data_good = std::make_shared<std::vector<T>>(1);
        auto data_bad  = std::make_shared<std::vector<T>>(2);
        REQUIRE_NOTHROW(ndarray<T, 1>({1}, data_good));
        REQUIRE_THROWS_AS((ndarray<T, 1>({1}, data_bad)), std::invalid_argument);
    }
}


TEST_CASE("ndarray can be created from basic factories", "[ndarray] [factories]")
{
    SECTION("1d arange works correctly")
    {
        auto A = ndarray<T, 1>::arange(10);
        auto x = T();

        REQUIRE(A.size() == 10);
        REQUIRE(A.shape()[0] == 10);

        for (const auto& a : A)
        {
            REQUIRE(a == x++);
        }
    }

    SECTION("2d arange works correctly")
    {
        auto A = ndarray<T, 2>::arange(10, 10);
        auto x = T();

        REQUIRE(A.size() == 100);
        REQUIRE(A.shape()[0] == 10);
        REQUIRE(A.shape()[1] == 10);

        for (const auto& a : A)
        {
            REQUIRE(a == x++);
        }
    }
}


TEST_CASE("ndarray selection works correctly", "[ndarray] [select]")
{
    auto A = ndarray<T, 2>(3, 4);
    auto B0 = A.select(std::make_tuple(0, 3), 0);
    auto B1 = A.select(0, std::make_tuple(0, 4));

    REQUIRE(B0.shape() == std::array<int, 1>{3});
    REQUIRE(B1.shape() == std::array<int, 1>{4});
    REQUIRE_FALSE(B0.contiguous());
    REQUIRE_FALSE(B1.contiguous());
}


TEST_CASE("ndarray can be serialized to and loaded from a string", "[ndarray] [serialize]")
{
    SECTION("ndarray can be serialized and loaded")
    {
        REQUIRE(ndarray<T, 1>::loads(ndarray<T, 1>::arange(10).dumps()).size() == 10);
        REQUIRE(ndarray<T, 2>::loads(ndarray<T, 2>::arange(10, 9).dumps()).size() == 10 * 9);
        REQUIRE(ndarray<T, 3>::loads(ndarray<T, 3>::arange(10, 9, 8).dumps()).size() == 10 * 9 * 8);
    }

    SECTION("narray throws if attempting to load from invalid string")
    {
        REQUIRE_THROWS_AS((ndarray<T, 1>::loads("")), std::invalid_argument);
        REQUIRE_THROWS_AS((ndarray<T, 1>::loads(ndarray<T, 1>::arange(10).dumps() + "1234")), std::invalid_argument);
        REQUIRE_THROWS_AS((ndarray<T, 1>::loads(ndarray<T, 1>::arange(10).dumps() + "12345678")), std::invalid_argument);
    }

    SECTION("ndarray dtype strings are as expected")
    {
        REQUIRE(ndarray<float,  1>::arange(10).dumps().substr(0, 2) == "f4");
        REQUIRE(ndarray<double, 1>::arange(10).dumps().substr(0, 2) == "f8");
        REQUIRE(ndarray<int,    1>::arange(10).dumps().substr(0, 2) == "i4");
        REQUIRE(ndarray<long,   1>::arange(10).dumps().substr(0, 2) == "i8");

        // Should not compile, native type without a type string:
        // REQUIRE(ndarray<char, 1>::arange(10).dumps().substr(0, 2) == "S0");
    }
}


#endif // TEST_NDARRAY