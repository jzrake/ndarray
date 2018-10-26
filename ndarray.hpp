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
template<typename Op, int Rank> class binary_operation;
template<int Rank> class ndarray;




// ============================================================================
template<typename Op, int Rank>
struct binary_op
{
    static ndarray<Rank> perform(const ndarray<Rank>& A, const ndarray<Rank>& B)
    {
        assert(A.shape() == B.shape());

        auto op = Op();
        auto C = ndarray<Rank>(A.shape());
        auto a = A.begin();
        auto b = B.begin();
        auto c = C.begin();

        for (; a != A.end(); ++a, ++b, ++c)
            *c = op(*a, *b);

        return C;
    }
    static void perform(ndarray<Rank>& A, const ndarray<Rank>& B)
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
template<int Rank>
class ndarray
{
public:


    enum { rank = Rank };


    /**
     * Constructors
     * 
     */
    // ========================================================================
    template<int R = rank, typename = typename std::enable_if<R == 0>::type>
    ndarray(double value=double())
    : scalar_offset(0)
    , data(std::make_shared<std::vector<double>>(1, value))
    {
    }

    template<int R = rank, typename = typename std::enable_if<R == 0>::type>
    ndarray(int scalar_offset, std::shared_ptr<std::vector<double>>& data)
    : scalar_offset(scalar_offset)
    , data(data)
    {
    }

    template<int R = rank, typename = typename std::enable_if<R == 1>::type>
    ndarray(std::initializer_list<double> elements)
    : count({int(elements.size())})
    , start({0})
    , final({int(elements.size())})
    , skips({1})
    , strides({1})
    , data(std::make_shared<std::vector<double>>(elements.begin(), elements.end()))
    {
    }

    template<typename... Dims>
    ndarray(Dims... dims) : ndarray(std::array<int, rank>({dims...}))
    {
        static_assert(sizeof...(dims) == rank,
          "Number of arguments to ndarray constructor must match rank");
    }

    template<typename SelectorType>
    ndarray(SelectorType selector, std::shared_ptr<std::vector<double>>& data)
    : count(selector.count)
    , start(selector.start)
    , final(selector.final)
    , skips(selector.skips)
    , strides(compute_strides(count))
    , data(data)
    {
    }

    ndarray() : ndarray(constant_array<int, rank>(0))
    {
    }

    ndarray(std::array<int, rank> dim_sizes)
    : count(dim_sizes)
    , start(constant_array<int, rank>(0))
    , final(dim_sizes)
    , skips(constant_array<int, rank>(1))
    , strides(compute_strides(count))
    , data(std::make_shared<std::vector<double>>(product(dim_sizes)))
    {
    }

    ndarray(std::array<int, rank> dim_sizes, std::shared_ptr<std::vector<double>>& data)
    : count(dim_sizes)
    , start(constant_array<int, rank>(0))
    , final(dim_sizes)
    , skips(constant_array<int, rank>(1))
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
        std::shared_ptr<std::vector<double>>& data)
    : count(count)
    , start(start)
    , final(final)
    , skips(constant_array<int, rank>(1))
    , strides(compute_strides(count))
    , data(data)
    {
        NDARRAY_ASSERT_VALID_ARGUMENT(
            data->size() == std::accumulate(count.begin(), count.end(), 1, std::multiplies<>()),
            "Size of data buffer is not the product of dim sizes");
    }

    ndarray(const ndarray<rank>& other)
    : count(other.shape())
    , start(constant_array<int, rank>(0))
    , final(other.shape())
    , skips(constant_array<int, rank>(1))
    , strides(compute_strides(count))
    , data(std::make_shared<std::vector<double>>(size()))
    {
        *this = other;
    }

    ndarray(ndarray<rank>& other)
    {
        become(other);
    }




    /**
     * Assignment operators
     * 
     */
    // ========================================================================
    ndarray<rank>& operator=(double value)
    {
        for (auto& a : *this)
        {
            a = value;
        }
        return *this;
    }

    ndarray<rank>& operator=(const ndarray<rank>& other)
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

    void become(ndarray<rank>& other)
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
    //     become(ndarray<rank>(sizes...));
    // }




    /**
     * Factories
     * 
     */
    // ========================================================================
    static ndarray<rank> stack(std::initializer_list<ndarray<rank - 1>> arrays)
    {
        if (arrays.size() == 0)
        {
            return ndarray<rank>();            
        }

        auto required_shape = arrays.begin()->shape();

        std::array<int, rank> dim_sizes;
        dim_sizes[0] = arrays.size();

        for (int n = 1; n < rank; ++n)
        {
            dim_sizes[n] = required_shape[n - 1];
        }
        auto A = ndarray<rank>(dim_sizes);
        int n = 0;

        for (const auto& array : arrays)
        {
            A[n] = array;
            ++n;
        }
        return A;
    }

    template<typename... Sizes>
    static ndarray<rank> arange(Sizes... sizes)
    {
        auto A = ndarray<rank>(sizes...);
        auto x = double();

        for (auto& a : A)
        {
            a = x++;
        }
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
    ndarray<rank - 1> operator[](int index)
    {
        return {offset_relative({index}), data};
    }

    template <int R = rank, typename std::enable_if_t<R == 1>* = nullptr>
    ndarray<rank - 1> operator[](int index) const
    {
        auto d = std::make_shared<std::vector<double>>(1, data->operator[](offset_relative({index})));
        return {0, d};
    }

    template <int R = rank, typename std::enable_if_t<R != 1>* = nullptr>
    ndarray<rank - 1> operator[](int index)
    {
        return {make_selector().select(index), data};
    }

    template <int R = rank, typename std::enable_if_t<R != 1>* = nullptr>
    ndarray<rank - 1> operator[](int index) const
    {
        auto S = make_selector().collapse(index);
        auto d = std::make_shared<std::vector<double>>(S.size());
        auto a = d->begin();
        auto b = begin();

        for ( ; a != d->end(); ++a, ++b)
        {
            *a = *b;
        }
        return {S, d};
    }

    template<typename... Index>
    double& operator()(Index... index)
    {
        static_assert(sizeof...(index) == rank,
          "Number of arguments to ndarray::operator() must match rank");

        return data->operator[](offset_relative({index...}));
    }

    template<typename... Index>
    const double& operator()(Index... index) const
    {
        static_assert(sizeof...(index) == rank,
          "Number of arguments to ndarray::operator() must match rank");

        return data->operator[](offset_relative({index...}));
    }

    template<typename... Index>
    auto select(Index... index)
    {
        auto S = make_selector().select(index...);
        return ndarray<S.rank>(S, data);
    }

    operator double() const
    {
        static_assert(rank == 0, "can only convert rank-0 array to scalar value");
        return data->operator[](scalar_offset);
    }

    bool is(const ndarray<rank>& other) const
    {
        return (scalar_offset == other.scalar_offset
        && count == other.count
        && start == other.start
        && final == other.final
        && skips == other.skips
        && strides == other.strides
        && data == other.data);
    }

    ndarray<rank> copy() const
    {
        auto d = std::make_shared<std::vector<double>>(begin(), end());
        return {shape(), d};
    }

    const std::vector<double>& container() const
    {
        return *data;
    }

    template<int other_rank>
    bool shares(const ndarray<other_rank>& other) const
    {
        return data == other.data;
    }




    // ========================================================================
    class iterator
    {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = double;
        using pointer = double*;
        using reference = double&;
        using iterator_category = std::forward_iterator_tag;

        iterator(ndarray<rank>& array, typename selector<rank>::iterator it) : array(array), it(it) {}
        iterator& operator++() { it.operator++(); return *this; }
        iterator operator++(int) { auto ret = *this; this->operator++(); return ret; }
        bool operator==(iterator other) const { return array.is(other.array) && it == other.it; }
        bool operator!=(iterator other) const { return array.is(other.array) && it != other.it; }
        double& operator*() { return array.data->operator[](array.offset_absolute(*it)); }
    private:
        typename selector<rank>::iterator it;
        ndarray<rank>& array;
    };

    iterator begin() { return {*this, make_selector().begin()}; }
    iterator end() { return {*this, make_selector().end()}; }




    // ========================================================================
    class const_iterator
    {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = const double;
        using pointer = const double*;
        using reference = const double&;
        using iterator_category = std::forward_iterator_tag;

        const_iterator(const ndarray<rank>& array, typename selector<rank>::iterator it) : array(array), it(it) {}
        const_iterator& operator++() { it.operator++(); return *this; }
        const_iterator operator++(int) { auto ret = *this; this->operator++(); return ret; }
        bool operator==(const_iterator other) const { return array.is(other.array) && it == other.it; }
        bool operator!=(const_iterator other) const { return array.is(other.array) && it != other.it; }
        const double& operator*() const { return array.data->operator[](array.offset_absolute(*it)); }
    private:
        typename selector<rank>::iterator it;
        const ndarray<rank>& array;
    };

    const_iterator begin() const { return {*this, make_selector().begin()}; }
    const_iterator end() const { return {*this, make_selector().end()}; }




    /**
     * Arithmetic operations
     * 
     */
    // ========================================================================
    ndarray<rank>& operator+=(double value) { for (auto& a : *this) { a += value; } return *this; }
    ndarray<rank>& operator-=(double value) { for (auto& a : *this) { a -= value; } return *this; }
    ndarray<rank>& operator*=(double value) { for (auto& a : *this) { a *= value; } return *this; }
    ndarray<rank>& operator/=(double value) { for (auto& a : *this) { a /= value; } return *this; }

    ndarray<rank>& operator+=(const ndarray<rank>& other) { binary_op<std::plus      <double>, rank>::perform(*this, other); return *this; }
    ndarray<rank>& operator-=(const ndarray<rank>& other) { binary_op<std::minus     <double>, rank>::perform(*this, other); return *this; }
    ndarray<rank>& operator*=(const ndarray<rank>& other) { binary_op<std::multiplies<double>, rank>::perform(*this, other); return *this; }
    ndarray<rank>& operator/=(const ndarray<rank>& other) { binary_op<std::divides   <double>, rank>::perform(*this, other); return *this; }

    ndarray<rank> operator+(double value) const { auto A = copy(); for (auto& a : A) { a += value; } return A; }
    ndarray<rank> operator-(double value) const { auto A = copy(); for (auto& a : A) { a -= value; } return A; }
    ndarray<rank> operator*(double value) const { auto A = copy(); for (auto& a : A) { a *= value; } return A; }
    ndarray<rank> operator/(double value) const { auto A = copy(); for (auto& a : A) { a /= value; } return A; }

    ndarray<rank> operator+(const ndarray<rank>& other) const { return binary_op<std::plus      <double>, rank>::perform(*this, other); }
    ndarray<rank> operator-(const ndarray<rank>& other) const { return binary_op<std::minus     <double>, rank>::perform(*this, other); }
    ndarray<rank> operator*(const ndarray<rank>& other) const { return binary_op<std::multiplies<double>, rank>::perform(*this, other); }
    ndarray<rank> operator/(const ndarray<rank>& other) const { return binary_op<std::divides   <double>, rank>::perform(*this, other); }




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

        using T = double;

        auto R = int(rank);
        auto S = shape();
        auto str = std::string();

        str.insert(str.end(), (char*)&R, (char*)(&R + 1));
        str.insert(str.end(), (char*)&S, (char*)(&S + 1));

        for (const auto& x : *this)
        {
            str.insert(str.end(), (char*)&x, (char*)(&x + 1));
        }
        return str;
    }

    static ndarray<rank> loads(const std::string& str)
    {
        using T = double;

        auto it = str.begin();
        auto data = std::make_shared<std::vector<T>>();
        auto x = T();
        auto R = int();
        auto S = constant_array<int, rank>(0);

        NDARRAY_ASSERT_VALID_ARGUMENT(it + sizeof(R) <= str.end(), "unexpected end of ndarray header string");
        std::memcpy(&R, &*it, sizeof(R));
        it += sizeof(R);

        NDARRAY_ASSERT_VALID_ARGUMENT(it + sizeof(S) <= str.end(), "unexpected end of ndarray header string");
        std::memcpy(&S, &*it, sizeof(S));
        it += sizeof(S);

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
            constant_array<int, rank>(0),
            count,
            constant_array<int, rank>(1)).strides();
    }

    template<typename T, int length>
    static std::array<T, length> constant_array(T value)
    {
        std::array<T, length> A;
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
    std::shared_ptr<std::vector<double>> data;




    /**
     * Grant friendship to ndarray's of other ranks.
     *
     */
    template<int other_rank>
    friend class ndarray;
    friend class iterator;
};




// ============================================================================
#ifdef TEST_NDARRAY
#include "catch.hpp"


TEST_CASE("ndarray can be constructed ", "[ndarray]")
{
    SECTION("trivial construction works OK")
    {
        REQUIRE(ndarray<1>(1).size() == 1);
        REQUIRE(ndarray<1>(1).shape() == std::array<int, 1>{1});
        REQUIRE(ndarray<1>().empty());
        REQUIRE_FALSE(ndarray<1>(1).empty());
    }

    SECTION("ndarray constructor throws if the data buffer has the wrong size")
    {
        auto data_good = std::make_shared<std::vector<double>>(1);
        auto data_bad  = std::make_shared<std::vector<double>>(2);
        REQUIRE_NOTHROW(ndarray<1>({1}, data_good));
        REQUIRE_THROWS_AS(ndarray<1>({1}, data_bad), std::invalid_argument);
    }
}


TEST_CASE("ndarray can be created from basic factories", "[ndarray] [factories]")
{
    SECTION("1d arange works correctly")
    {
        auto A = ndarray<1>::arange(10);
        auto x = double();

        REQUIRE(A.size() == 10);
        REQUIRE(A.shape()[0] == 10);

        for (const auto& a : A)
        {
            REQUIRE(a == x++);
        }
    }

    SECTION("2d arange works correctly")
    {
        auto A = ndarray<2>::arange(10, 10);
        auto x = double();

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
    auto A = ndarray<2>(3, 4);
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
        REQUIRE(ndarray<1>::loads(ndarray<1>::arange(10).dumps()).size() == 10);
        REQUIRE(ndarray<2>::loads(ndarray<2>::arange(10, 9).dumps()).size() == 10 * 9);
        REQUIRE(ndarray<3>::loads(ndarray<3>::arange(10, 9, 8).dumps()).size() == 10 * 9 * 8);
    }

    SECTION("narray throws if attempting to load from invalid string")
    {
        REQUIRE_THROWS_AS(ndarray<1>::loads(""), std::invalid_argument);
        REQUIRE_THROWS_AS(ndarray<1>::loads(ndarray<1>::arange(10).dumps() + "1234"), std::invalid_argument);
        REQUIRE_THROWS_AS(ndarray<1>::loads(ndarray<1>::arange(10).dumps() + "12345678"), std::invalid_argument);
    }
}


#endif // TEST_NDARRAY