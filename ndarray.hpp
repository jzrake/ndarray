#pragma once
#include <array>
#include <memory>
#include <numeric>
#include "shape.hpp"
#include "selector.hpp"
#include "buffer.hpp"




// ============================================================================
namespace nd // ND_API_START
{
    template<typename T, typename U, int R, typename Op> class binary_op;
    template<typename T, int R, typename Op> class unary_op;
    template<typename T, int R> class ndarray;
    template<typename T> struct dtype_str;

    template<typename T> ndarray<T, 1> arange(int size);
    template<typename T> ndarray<T, 1> ones(int size);
    template<typename T> ndarray<T, 1> zeros(int size);

    template<typename T, int R>
    nd::ndarray<T, R + 1> stack(std::initializer_list<nd::ndarray<T, R - 1>> arrays);
} // ND_API_END




// ============================================================================
template<typename T> nd::ndarray<T, 1> nd::arange(int size) // ND_IMPL_START
{
    auto A = nd::ndarray<T, 1>(size);
    auto x = T();
    for (auto& a : A) a = x++;
    return A;
}

template<typename T> nd::ndarray<T, 1> nd::ones(int size)
{
    auto A = nd::ndarray<T, 1>(size);
    for (auto& a : A) a = 1;
    return A;
}

template<typename T> nd::ndarray<T, 1> nd::zeros(int size)
{
    auto A = nd::ndarray<T, 1>(size);
    for (auto& a : A) a = 0;
    return A;
}

template<typename T, int R> /* UNTESTED */
nd::ndarray<T, R + 1> nd::stack(std::initializer_list<nd::ndarray<T, R - 1>> arrays)
{
    if (arrays.size() == 0)
    {
        return nd::ndarray<T, R>();            
    }

    auto required_shape = arrays.begin()->shape();

    std::array<int, R> dim_sizes;
    dim_sizes[0] = arrays.size();

    for (int n = 1; n < R; ++n)
    {
        dim_sizes[n] = required_shape[n - 1];
    }
    auto A = nd::ndarray<T, R>(dim_sizes);
    auto n = 0;

    for (const auto& array : arrays)
    {
        A[n] = array;
        ++n;
    }
    return A;
}




// ============================================================================
template<typename T, int R, typename Op>
struct nd::unary_op
{
    static auto perform(const ndarray<T, R>& A)
    {
        auto op = Op();
        auto B = ndarray<decltype(op(T())), R>(A.shape());
        auto a = A.begin();
        auto b = B.begin();

        for (; a != A.end(); ++a, ++b)
            *b = op(*a);

        return B;
    }
};




// ============================================================================
template<typename T, typename U, int R, typename Op>
struct nd::binary_op
{
    static auto perform(const ndarray<T, R>& A, const ndarray<U, R>& B)
    {
        if (A.shape() != B.shape())
            throw std::invalid_argument("incompatible shapes for binary operation");

        auto op = Op();
        auto C = ndarray<decltype(op(T(), U())), R>(A.shape());
        auto a = A.begin();
        auto b = B.begin();
        auto c = C.begin();

        for (; a != A.end(); ++a, ++b, ++c)
            *c = op(*a, *b);

        return C;
    }

    static auto perform(const ndarray<T, R>& A, U b)
    {
        auto op = Op();
        auto C = ndarray<decltype(op(T(), U())), R>(A.shape());
        auto a = A.begin();
        auto c = C.begin();

        for (; a != A.end(); ++a, ++c)
            *c = op(*a, b);

        return C;
    }

    static void perform(ndarray<T, R>& A, const ndarray<U, R>& B)
    {
        if (A.shape() != B.shape())
            throw std::invalid_argument("incompatible shapes for binary operation");

        auto op = Op();
        auto a = A.begin();
        auto b = B.begin();

        for (; a != A.end(); ++a, ++b)
            *a = op(*a, *b);
    }
};




// ============================================================================
template<> struct nd::dtype_str<bool  > { static std::array<char, 8> value; };
template<> struct nd::dtype_str<float > { static std::array<char, 8> value; };
template<> struct nd::dtype_str<double> { static std::array<char, 8> value; };
template<> struct nd::dtype_str<int   > { static std::array<char, 8> value; };
template<> struct nd::dtype_str<long  > { static std::array<char, 8> value; };

std::array<char, 8> nd::dtype_str<bool  >::value = {'b','1',  0,  0,  0,  0,  0,  0};
std::array<char, 8> nd::dtype_str<float >::value = {'f','4',  0,  0,  0,  0,  0,  0};
std::array<char, 8> nd::dtype_str<double>::value = {'f','8',  0,  0,  0,  0,  0,  0};
std::array<char, 8> nd::dtype_str<int   >::value = {'i','4',  0,  0,  0,  0,  0,  0};
std::array<char, 8> nd::dtype_str<long  >::value = {'i','8',  0,  0,  0,  0,  0,  0};




// ============================================================================
template<typename T, int R>
class nd::ndarray
{
public:


    using dtype = T;
    enum { rank = R };


    /**
     * Constructors
     * 
     */
    // ========================================================================
    template<int Rank = R, typename = typename std::enable_if<Rank == 0>::type>
    ndarray(T value=T())
    : scalar_offset(0)
    , buf(std::make_shared<buffer<T>>(1, value))
    {
    }

    template<int Rank = R, typename = typename std::enable_if<Rank == 0>::type>
    ndarray(int scalar_offset, std::shared_ptr<buffer<T>>& buf)
    : scalar_offset(scalar_offset)
    , buf(buf)
    {
    }

    template<int Rank = R, typename = typename std::enable_if<Rank == 1>::type>
    ndarray(std::initializer_list<T> elements)
    : sel({elements.size()})
    , buf(std::make_shared<buffer<T>>(elements.begin(), elements.end()))
    , strides(sel.strides())
    {
    }

    template<typename... Dims>
    ndarray(Dims... dims) : ndarray(std::array<int, R>({dims...}))
    {
        static_assert(sizeof...(dims) == rank,
          "Number of arguments to ndarray constructor must match rank");
    }

    template<typename SelectorType>
    ndarray(SelectorType sel, std::shared_ptr<buffer<T>>& buf)
    : sel(sel)
    , buf(buf)
    , strides(sel.strides())
    {
    }

    ndarray() : ndarray(constant_array<rank>(0))
    {
    }

    ndarray(std::array<int, R> dim_sizes)
    : sel(dim_sizes)
    , strides(sel.strides())
    , buf(std::make_shared<buffer<T>>(sel.size()))
    {
    }

    ndarray(std::array<int, R> dim_sizes, std::shared_ptr<buffer<T>>& buf)
    : sel(dim_sizes)
    , buf(buf)
    , strides(sel.strides())
    {
        assert_valid_argument(buf->size() == sel.size(),
            "Size of data buffer is not the product of dim sizes");
    }

    ndarray(const ndarray<T, R>& other)
    : sel(other.sel.shape())
    , strides(sel.strides())
    , buf(std::make_shared<buffer<T>>(size()))
    {
        *this = other;
    }

    ndarray(ndarray<T, R>& other)
    {
        become(other);
    }




    /**
     * Assignment operators
     * 
     */
    // ========================================================================
    template <int Rank = R, typename std::enable_if_t<Rank == 0>* = nullptr>
    ndarray<T, R>& operator=(T value)
    {
        buf->operator[](scalar_offset) = value;
        return *this;
    }

    template <int Rank = R, typename std::enable_if_t<Rank != 0>* = nullptr>
    ndarray<T, R>& operator=(T value)
    {
        for (auto& a : *this)
            a = value;

        return *this;
    }

    ndarray<T, R>& operator=(const ndarray<T, R>& other)
    {
        assert_valid_argument(shape() == other.shape(),
            "assignment from ndarray of incompatible shape");

        copy_internal(*this, other);
        return *this;
    }

    void become(ndarray<T, R>& other)
    {
        strides = other.strides;
        sel = other.sel;
        buf = other.buf;
    }

    template<typename... Sizes>
    auto reshape(Sizes... sizes)
    {
        if (! contiguous())
        {
            auto A = ndarray<T, sizeof...(Sizes)>(sizes...);
            copy_internal(A, *this);
            return A;
        }
        return ndarray<T, sizeof...(Sizes)>({sizes...}, buf);
    }

    template<typename... Sizes>
    const auto reshape(Sizes... sizes) const
    {
        if (! contiguous())
        {
            auto A = ndarray<T, sizeof...(Sizes)>(sizes...);
            copy_internal(A, *this);
            return A;
        }
        return ndarray<T, sizeof...(Sizes)>({sizes...}, const_cast<std::shared_ptr<buffer<T>>&>(buf));
        // auto A = ndarray<T, sizeof...(Sizes)>(sizes...);
        // copy_internal(A, *this);
        // return A;
    }




    /**
     * Shape and size query methods
     * 
     */
    // ========================================================================
    auto size() const { return sel.size(); }
    auto shape() const { return sel.shape(); }
    bool empty() const { return sel.empty(); }
    bool contiguous() const { return sel.contiguous(); }




    /**
     * Data accessors and selection methods
     * 
     */
    // ========================================================================
    template <int Rank = R, typename std::enable_if_t<Rank == 1>* = nullptr>
    ndarray<T, R - 1> operator[](int index)
    {
        if (index < 0 || index >= (sel.final[0] - sel.start[0]) / sel.skips[0])
            throw std::out_of_range("ndarray: index out of range");

        return {offset_relative({index}), buf};
    }

    template <int Rank = R, typename std::enable_if_t<Rank == 1>* = nullptr>
    const ndarray<T, R - 1> operator[](int index) const
    {
        if (index < 0 || index >= (sel.final[0] - sel.start[0]) / sel.skips[0])
            throw std::out_of_range("ndarray: index out of range");

        return {offset_relative({index}), const_cast<std::shared_ptr<buffer<T>>&>(buf)};

        // auto d = std::make_shared<buffer<T>>(1, buf->operator[](offset_relative({index})));
        // return {0, d};
    }

    template <int Rank = R, typename std::enable_if_t<Rank != 1>* = nullptr>
    ndarray<T, R - 1> operator[](int index)
    {
        if (index < 0 || index >= (sel.final[0] - sel.start[0]) / sel.skips[0])
            throw std::out_of_range("ndarray: index out of range");

        return {sel.select(index), buf};
    }

    template <int Rank = R, typename std::enable_if_t<Rank != 1>* = nullptr>
    const ndarray<T, R - 1> operator[](int index) const
    {
        if (index < 0 || index >= (sel.final[0] - sel.start[0]) / sel.skips[0])
            throw std::out_of_range("ndarray: index out of range");

        return {sel.select(index), const_cast<std::shared_ptr<buffer<T>>&>(buf)};

        // auto S = sel.select(index);
        // auto d = std::make_shared<buffer<T>>(S.size());
        // auto a = d->begin();
        // auto b = begin();

        // for ( ; a != d->end(); ++a, ++b)
        //     *a = *b;

        // return {S, d};
    }

    template<typename... Index>
    T& operator()(Index... index)
    {
        if (! sel.contains(index...))
            throw std::out_of_range("ndarray: index out of range");

        return buf->operator[](offset_relative({index...}));
    }

    template<typename... Index>
    const T& operator()(Index... index) const
    {
        if (! sel.contains(index...))
            throw std::out_of_range("ndarray: selection out of range");

        return buf->operator[](offset_relative({index...}));
    }

    template<typename... Index>
    auto select(Index... index)
    {
        if (! sel.contains(index...))
            throw std::out_of_range("ndarray: selection out of range");

        auto S = sel.select(index...);
        return ndarray<T, S.rank>(S.reset(), buf);
    }

    template<typename... Index>
    const auto select(Index... index) const
    {
        if (! sel.contains(index...))
            throw std::out_of_range("ndarray: selection out of range");

        auto S = sel.select(index...);
        return ndarray<T, S.rank>(S.reset(), const_cast<std::shared_ptr<buffer<T>>&>(buf));

        // auto S = sel.select(index...);
        // auto d = std::make_shared<buffer<T>>(S.size());
        // auto a = d->begin();
        // auto b = begin();

        // for ( ; a != d->end(); ++a, ++b)
        //     *a = *b;

        // return ndarray<T, S.rank>(S.reset(), d);
    }

    operator T() const
    {
        static_assert(rank == 0, "can only convert rank-0 array to scalar value");
        return buf->operator[](scalar_offset);
    }

    ndarray<T, R> copy() const
    {
        auto d = std::make_shared<buffer<T>>(begin(), end());
        return {shape(), d};
    }

    template<typename new_type>
    ndarray<new_type, R> astype() const
    {
        auto d = std::make_shared<buffer<new_type>>(begin(), end());
        return {shape(), d};
    }

    const T* data() const
    {
        return buf->data();
    }




    /**
     * Operations
     * 
     */
    // ========================================================================
    template<typename U> struct OpEquals     { auto operator()(T a, U b) const { return a == b; } };
    template<typename U> struct OpNotEquals  { auto operator()(T a, U b) const { return a != b; } };
    template<typename U> struct OpGreaterEq  { auto operator()(T a, U b) const { return a >= b; } };
    template<typename U> struct OpLessEq     { auto operator()(T a, U b) const { return a <= b; } };
    template<typename U> struct OpGreater    { auto operator()(T a, U b) const { return a > b; } };
    template<typename U> struct OpLess       { auto operator()(T a, U b) const { return a < b; } };
    template<typename U> struct OpPlus       { auto operator()(T a, U b) const { return a + b; } };
    template<typename U> struct OpMinus      { auto operator()(T a, U b) const { return a - b; } };
    template<typename U> struct OpMultiplies { auto operator()(T a, U b) const { return a * b; } };
    template<typename U> struct OpDivides    { auto operator()(T a, U b) const { return a / b; } };
    struct OpIdentity { auto operator()(T a) const { return a; } };
    struct OpNegate   { auto operator()(T a) const { return ! a; } };




    /**
     * Arithmetic operations
     * 
     */
    // ========================================================================
    template<typename U> auto& operator+=(U b) { for (auto& a : *this) { a += b; } return *this; }
    template<typename U> auto& operator-=(U b) { for (auto& a : *this) { a -= b; } return *this; }
    template<typename U> auto& operator*=(U b) { for (auto& a : *this) { a *= b; } return *this; }
    template<typename U> auto& operator/=(U b) { for (auto& a : *this) { a /= b; } return *this; }
    template<typename U> auto& operator+=(const ndarray<U, R>& B) { binary_op<T, U, R, OpPlus      <U>>::perform(*this, B); return *this; }
    template<typename U> auto& operator-=(const ndarray<U, R>& B) { binary_op<T, U, R, OpMinus     <U>>::perform(*this, B); return *this; }
    template<typename U> auto& operator*=(const ndarray<U, R>& B) { binary_op<T, U, R, OpMultiplies<U>>::perform(*this, B); return *this; }
    template<typename U> auto& operator/=(const ndarray<U, R>& B) { binary_op<T, U, R, OpDivides   <U>>::perform(*this, B); return *this; }

    template<typename U> auto operator+(U b) const { auto A = copy(); for (auto& a : A) { a += b; } return A; }
    template<typename U> auto operator-(U b) const { auto A = copy(); for (auto& a : A) { a -= b; } return A; }
    template<typename U> auto operator*(U b) const { auto A = copy(); for (auto& a : A) { a *= b; } return A; }
    template<typename U> auto operator/(U b) const { auto A = copy(); for (auto& a : A) { a /= b; } return A; }
    template<typename U> auto operator+(const ndarray<U, R>& B) const { return binary_op<T, U, R, OpPlus      <U>>::perform(*this, B); }
    template<typename U> auto operator-(const ndarray<U, R>& B) const { return binary_op<T, U, R, OpMinus     <U>>::perform(*this, B); }
    template<typename U> auto operator*(const ndarray<U, R>& B) const { return binary_op<T, U, R, OpMultiplies<U>>::perform(*this, B); }
    template<typename U> auto operator/(const ndarray<U, R>& B) const { return binary_op<T, U, R, OpDivides   <U>>::perform(*this, B); }




    /**
     * Comparison/equality methods and operators
     * 
     */
    // ========================================================================
    template<typename U> auto operator==(const ndarray<U, R>& B) const { return binary_op<T, U, R, OpEquals   <U>>::perform(*this, B); }
    template<typename U> auto operator!=(const ndarray<U, R>& B) const { return binary_op<T, U, R, OpNotEquals<U>>::perform(*this, B); }
    template<typename U> auto operator>=(const ndarray<U, R>& B) const { return binary_op<T, U, R, OpGreaterEq<U>>::perform(*this, B); }
    template<typename U> auto operator<=(const ndarray<U, R>& B) const { return binary_op<T, U, R, OpLessEq   <U>>::perform(*this, B); }
    template<typename U> auto operator> (const ndarray<U, R>& B) const { return binary_op<T, U, R, OpGreater  <U>>::perform(*this, B); }
    template<typename U> auto operator< (const ndarray<U, R>& B) const { return binary_op<T, U, R, OpLess     <U>>::perform(*this, B); }

    template<typename U> auto operator==(U b) const { return binary_op<T, U, R, OpEquals   <U>>::perform(*this, b); }
    template<typename U> auto operator!=(U b) const { return binary_op<T, U, R, OpNotEquals<U>>::perform(*this, b); }
    template<typename U> auto operator>=(U b) const { return binary_op<T, U, R, OpGreaterEq<U>>::perform(*this, b); }
    template<typename U> auto operator<=(U b) const { return binary_op<T, U, R, OpLessEq   <U>>::perform(*this, b); }
    template<typename U> auto operator> (U b) const { return binary_op<T, U, R, OpGreater  <U>>::perform(*this, b); }
    template<typename U> auto operator< (U b) const { return binary_op<T, U, R, OpLess     <U>>::perform(*this, b); }

    auto operator!() const { return unary_op<T, R, OpNegate>::perform(*this); }
    bool any() const { for (auto x : *this) if (x) return true; return false; }
    bool all() const { for (auto x : *this) if (! x) return false; return true; }

    bool is(const ndarray<T, R>& other) const
    {
        return (scalar_offset == other.scalar_offset
        && strides == other.strides
        && sel == other.sel
        && buf == other.buf);
    }

    template<int other_rank>
    bool shares(const ndarray<T, other_rank>& other) const
    {
        return buf == other.buf;
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

        iterator(ndarray<T, R>& array, typename selector<rank>::iterator it) : array(array), it(it) {}
        iterator& operator++() { it.operator++(); return *this; }
        iterator operator++(int) { auto ret = *this; this->operator++(); return ret; }
        bool operator==(iterator other) const { return array.is(other.array) && it == other.it; }
        bool operator!=(iterator other) const { return array.is(other.array) && it != other.it; }
        T& operator*() { return array.buf->operator[](array.offset_absolute(*it)); }

        auto index() const { return *it; }
        ndarray<T, R>& get_array() const { return array; }

    private:
        ndarray<T, R>& array;
        typename selector<rank>::iterator it;
    };

    iterator begin() { static_assert(R > 0, "cannot iterate over scalar"); return {*this, sel.begin()}; }
    iterator end()   { static_assert(R > 0, "cannot iterate over scalar"); return {*this, sel.end()}; }




    // ========================================================================
    class const_iterator
    {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = const T;
        using pointer = const T*;
        using reference = const T&;
        using iterator_category = std::forward_iterator_tag;

        const_iterator(const ndarray<T, R>& array, typename selector<rank>::iterator it) : array(array), it(it) {}
        const_iterator& operator++() { it.operator++(); return *this; }
        const_iterator operator++(int) { auto ret = *this; this->operator++(); return ret; }
        bool operator==(const_iterator other) const { return array.is(other.array) && it == other.it; }
        bool operator!=(const_iterator other) const { return array.is(other.array) && it != other.it; }
        const T& operator*() const { return array.buf->operator[](array.offset_absolute(*it)); }

    private:
        const ndarray<T, R>& array;
        typename selector<rank>::iterator it;
    };

    const_iterator begin() const { static_assert(R > 0, "cannot iterate over scalar"); return {*this, sel.begin()}; }
    const_iterator end()   const { static_assert(R > 0, "cannot iterate over scalar"); return {*this, sel.end()}; }




    /**
     * Basic serialization operations
     * 
     */
    // ========================================================================
    std::string dumps() const
    {
        // dtype ... 8 char's
        // rank  ... 1 int
        // shape ... rank int's
        // data  ... size T's

        auto D = dtype_str<T>::value;
        auto Q = int(rank);
        auto S = shape();
        auto str = std::string();

        str.insert(str.end(), (char*)&D, (char*)(&D + 1));
        str.insert(str.end(), (char*)&Q, (char*)(&Q + 1));
        str.insert(str.end(), (char*)&S, (char*)(&S + 1));

        for (const auto& x : *this)
        {
            str.insert(str.end(), (char*)&x, (char*)(&x + 1));
        }
        return str;
    }

    static ndarray<T, R> loads(const std::string& str)
    {
        auto it = str.begin();
        auto D = std::array<char, 8>();
        auto Q = int();
        auto S = constant_array<rank>(0);
        auto x = T();

        assert_valid_argument(it + sizeof(D) <= str.end(), "unexpected end of ndarray header string");
        std::memcpy(&D, &*it, sizeof(D));
        it += sizeof(D);

        assert_valid_argument(it + sizeof(Q) <= str.end(), "unexpected end of ndarray header string");
        std::memcpy(&Q, &*it, sizeof(Q));
        it += sizeof(Q);

        assert_valid_argument(it + sizeof(S) <= str.end(), "unexpected end of ndarray header string");
        std::memcpy(&S, &*it, sizeof(S));
        it += sizeof(S);

        assert_valid_argument(D == dtype_str<T>::value, "ndarray string has wrong data type");
        assert_valid_argument(Q == rank, "ndarray string has the wrong rank");

        auto size = std::accumulate(S.begin(), S.end(), 1, std::multiplies<>());
        auto wbuf = std::make_shared<buffer<T>>(size);
        auto dest = wbuf->begin();

        while (it != str.end())
        {
            assert_valid_argument(dest != wbuf->end(), "unexpected end of ndarray data string");
            std::memcpy(&*dest, &*it, sizeof(T));
            ++dest;
            it += sizeof(T);
        }

        return {S, wbuf};
    }


private:
    /**
     * Private utility methods
     * 
     */
    // ========================================================================
    int offset_relative(std::array<int, R> index) const
    {
        int m = scalar_offset;

        for (int n = 0; n < rank; ++n)
        {
            m += (sel.start[n] + sel.skips[n] * index[n]) * strides[n];
        }
        return m;
    }

    int offset_absolute(std::array<int, R> index) const
    {
        int m = scalar_offset;

        for (int n = 0; n < rank; ++n)
        {
            m += index[n] * strides[n];
        }
        return m;
    }

    template<int length>
    static std::array<int, length> constant_array(T value)
    {
        std::array<int, length> A;
        for (auto& a : A) a = value;
        return A;
    }

    static void assert_valid_argument(bool condition, const char* message)
    {
        if (! condition)
            throw std::invalid_argument(message);
    }

    template<typename target_type, int target_rank>
    static void copy_internal(ndarray<target_type, target_rank>& target, const ndarray<T, R>& source)
    {
        if (target.size() != source.size())
            throw std::invalid_argument("source and target arrays have different sizes");

        auto a = target.begin();
        auto b = source.begin();

        for (; a != target.end(); ++a, ++b)
            *a = *b;
    }



    /**
     * Data members
     *
     */
    // ========================================================================
    int scalar_offset = 0;
    selector<R> sel;
    std::array<int, R> strides;
    std::shared_ptr<buffer<T>> buf;




    /**
     * Grant friendship to ndarray's of other ranks.
     *
     */
    template<typename, int>
    friend class ndarray;
    friend class iterator;
}; // ND_IMPL_END




// ============================================================================
#ifdef TEST_NDARRAY
#include "catch.hpp"
using T = double;


TEST_CASE("ndarray can be constructed", "[ndarray]")
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
        auto data_good = std::make_shared<buffer<T>>(1);
        auto data_bad  = std::make_shared<buffer<T>>(2);
        REQUIRE_NOTHROW(ndarray<T, 1>({1}, data_good));
        REQUIRE_THROWS_AS((ndarray<T, 1>({1}, data_bad)), std::invalid_argument);
    }
}


TEST_CASE("ndarray can be created from basic factories", "[ndarray] [factories]")
{
    SECTION("arange works correctly")
    {
        auto A = nd::arange<double>(10);
        auto x = T();

        REQUIRE(A.size() == 10);
        REQUIRE(A.shape()[0] == 10);

        for (const auto& a : A)
        {
            REQUIRE(a == x++);
        }
    }

    SECTION("ones works correctly")
    {
        auto A = nd::ones<double>(10);

        REQUIRE(A.size() == 10);
        REQUIRE(A.shape()[0] == 10);

        for (const auto& a : A)
        {
            REQUIRE(a == 1);
        }
    }
}


TEST_CASE("ndarray can be copied and casted", "[ndarray]")
{
    auto A = nd::arange<double>(10);
    auto B = nd::arange<float>(10);
    auto C = A.astype<float>();
    auto D = A;
    const auto E = A;
    auto F = E;

    REQUIRE(B(0) == C(0));
    REQUIRE(B(1) == C(1));
    REQUIRE_FALSE(A.copy().shares(A));
    REQUIRE_FALSE(C.shares(B));
    REQUIRE(D.shares(A));
    REQUIRE(E.shares(A));
    REQUIRE_FALSE(F.shares(A));
}


TEST_CASE("ndarray leading axis slicing via operator[] works correctly", "[ndarray]")
{
    auto _ = nd::axis::all();

    SECTION("Works for non-const ndarray")
    {
        auto A = nd::ndarray<int, 3>(10, 12, 14);
        REQUIRE(A[0].shape() == std::array<int, 2>{12, 14});
        REQUIRE(A[0].shares(A));
        REQUIRE(A.select(_|0|10, _|0|12, _|0|14).shares(A));
    }

    SECTION("Works for const ndarray")
    {
        const auto A = nd::ndarray<int, 3>(10, 12, 14);
        REQUIRE(A[0].shape() == std::array<int, 2>{12, 14});
        REQUIRE(A[0].shares(A));
        REQUIRE(A.select(_|0|10, _|0|12, _|0|14).shares(A));

        // Should fail to compile:
        // A.select(_|0|10, _|0|12, _|0|14) = 1.0;
        // A.reshape(120, 14) = 10;
        // A.select(_|0|10, _|0|12, _|0|14) = 1.0;
    }

    SECTION("Can assign to returned by ndarray<1>::operator[]")
    {
        auto A = nd::ndarray<int, 1>(2);
        A[0] = 1;
        A[1] = 2;
        CHECK(A(0) == 1);
        CHECK(A(1) == 2);
    }

    SECTION("Can assign to returned by ndarray<2>::operator[]")
    {
        auto A = nd::ndarray<int, 2>(2, 3);
        A[0] = 1;
        A[1] = 2;
        CHECK(A(0, 0) == 1);
        CHECK(A(1, 0) == 2);
        CHECK((A[0] == 1).all());
        CHECK((A[1] == 2).all());
    }

    SECTION("Can assign to returned by ndarray<3>::operator[]")
    {
        auto A = nd::ndarray<int, 3>(2, 3, 4);
        A[0] = 1;
        A[1] = 2;
        CHECK(A(0, 0, 0) == 1);
        CHECK(A(1, 0, 0) == 2);
        CHECK((A[0] == 1).all());
        CHECK((A[1] == 2).all());
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


TEST_CASE("ndarray can return a reshaped version of itself", "[ndarray] [reshape]")
{
    auto A = nd::arange<T>(100);
    const auto B = nd::arange<T>(100);

    REQUIRE(A.reshape(10, 10).shape() == std::array<int, 2>{10, 10});
    REQUIRE(B.reshape(10, 10).shape() == std::array<int, 2>{10, 10});
    REQUIRE(A.reshape(10, 10).shares(A));
    REQUIRE(B.reshape(10, 10).shares(B));
    REQUIRE_THROWS_AS(A.reshape(10, 11), std::invalid_argument);
}


TEST_CASE("ndarray can be serialized to and loaded from a string", "[ndarray] [serialize] [safety]")
{
    SECTION("ndarray can be serialized and loaded")
    {
        REQUIRE(ndarray<T, 1>::loads(nd::arange<T>(10).dumps()).size() == 10);
        REQUIRE(ndarray<T, 2>::loads(nd::arange<T>(90).reshape(10, 9).dumps()).size() == 10 * 9);
    }

    SECTION("narray throws if attempting to load from invalid string")
    {
        REQUIRE_THROWS_AS((ndarray<T, 1>::loads("")), std::invalid_argument);
        REQUIRE_THROWS_AS((ndarray<T, 1>::loads(nd::arange<T>(10).dumps() + "1234")), std::invalid_argument);
        REQUIRE_THROWS_AS((ndarray<T, 1>::loads(nd::arange<T>(10).dumps() + "12345678")), std::invalid_argument);
    }

    SECTION("ndarray dtype strings are as expected")
    {
        REQUIRE(arange<float >(10).dumps().substr(0, 2) == "f4");
        REQUIRE(arange<double>(10).dumps().substr(0, 2) == "f8");
        REQUIRE(arange<int   >(10).dumps().substr(0, 2) == "i4");
        REQUIRE(arange<long  >(10).dumps().substr(0, 2) == "i8");

        // Should not compile, native type with no defined type string:
        // REQUIRE(ndarray<char, 1>::arange(10).dumps().substr(0, 2) == "S0");
    }
}


TEST_CASE("ndarray throws attempting to index out-of-bounds", "[ndarray] [safety]")
{
    auto _ = nd::axis::all();

    SECTION("operator() throws if out of bounds")
    {
        REQUIRE_THROWS_AS((ndarray<T, 1>(10)(-1)), std::out_of_range);
        REQUIRE_THROWS_AS((ndarray<T, 1>(10)(10)), std::out_of_range);
        REQUIRE_THROWS_AS((ndarray<T, 1>(10).select(_|0| 5)(-1)), std::out_of_range);
        REQUIRE_THROWS_AS((ndarray<T, 1>(10).select(_|0| 5)( 5)), std::out_of_range);
        REQUIRE_THROWS_AS((ndarray<T, 1>(10).select(_|5|10)(-1)), std::out_of_range);
        REQUIRE_THROWS_AS((ndarray<T, 1>(10).select(_|5|10)( 5)), std::out_of_range);

        REQUIRE_THROWS_AS((ndarray<T, 2>(10, 8).select(_|0| 5, _|0|8)(-1, 0)), std::out_of_range);
        REQUIRE_THROWS_AS((ndarray<T, 2>(10, 8).select(_|0| 5, _|0|8)( 5, 0)), std::out_of_range);
        REQUIRE_THROWS_AS((ndarray<T, 2>(10, 8).select(_|5|10, _|0|8)(-1, 0)), std::out_of_range);
        REQUIRE_THROWS_AS((ndarray<T, 2>(10, 8).select(_|5|10, _|0|8)( 5, 0)), std::out_of_range);

        REQUIRE_NOTHROW((ndarray<T, 1>(10).select(_|0|10|2)(0)));
        REQUIRE_NOTHROW((ndarray<T, 1>(10).select(_|0|10|2)(4)));
        REQUIRE_THROWS_AS((ndarray<T, 1>(10).select(_|0|10|2)(-1)), std::out_of_range);
        REQUIRE_THROWS_AS((ndarray<T, 1>(10).select(_|0|10|2)( 5)), std::out_of_range);
    }

    SECTION("operator[] throws if out of bounds")
    {
        REQUIRE_THROWS_AS((ndarray<T, 1>(10)[-1]), std::out_of_range);
        REQUIRE_THROWS_AS((ndarray<T, 1>(10)[10]), std::out_of_range);
        REQUIRE_THROWS_AS((ndarray<T, 2>(10, 8)[-1]), std::out_of_range);
        REQUIRE_THROWS_AS((ndarray<T, 2>(10, 8)[10]), std::out_of_range);
    }
}


TEST_CASE("ndarrays iterators respect const correctness", "[ndarray] [iterator]")
{
    SECTION("non-const ndarray iterator can be assigned to properly")
    {
        auto A = ndarray<double, 1>(10);
        auto it = A.begin();
        *it = 12;
        REQUIRE(*it == 12);
    }

    SECTION("const ndarray iterator cannot be assigned to")
    {
        const auto A = ndarray<double, 1>(10);
        auto it = A.begin();
        // *it = 12; /* should not compile! */
        REQUIRE_FALSE(*it == 12);
    }
}


TEST_CASE("ndarrays can be compared, evaluating to a boolean array", "[ndarray] [comparison]")
{
    SECTION("basic comparison of two arrays with different types works")
    {
        auto A = nd::arange<int>(10);
        auto B = nd::ones<int>(10);

        REQUIRE((A == A).all());
        REQUIRE((A == B).any());
        REQUIRE_FALSE((A == B).all());
    }
}


TEST_CASE("ndarrays can perform basic arithmetic", "[ndarray] [arithmetic]")
{
    SECTION("add, sub, mul, div of arrays with the same type works as expected")
    {
        auto A = nd::zeros<int>(10);
        auto B = nd::ones<int>(10);

        REQUIRE((A + 1 == B).all());
        REQUIRE((A - 1 == B - 2).all());
        REQUIRE_FALSE((A - 1 == B + 2).any());
    }

    SECTION("add, sub, mul, div of arrays with the different types works as expected")
    {
        auto A = nd::zeros<int>(10);
        auto B = nd::ones<double>(10);

        REQUIRE((A + 1 == B).all());
        REQUIRE((A - 1 == B - 2).all());
        REQUIRE_FALSE((A - 1 == B + 2).any());
    }
}


TEST_CASE("ndarrays can perform skipped assignments", "[ndarray]")
{
    auto _ = nd::axis::all();

    SECTION("skipped assignment to ndarray<1> works as expected")
    {
        auto A = nd::ndarray<int, 1>(9);
        A.select(_|0|9|3) = 1;
        A.select(_|1|9|3) = 2;
        A.select(_|2|9|3) = 3;
        REQUIRE(A.select(_|0|9|3).size() == 3);
        REQUIRE(A.select(_|1|9|3).size() == 3);
        REQUIRE(A.select(_|2|9|3).size() == 3);
        REQUIRE((A.select(_|0|9|3) == 1).all());
        REQUIRE((A.select(_|1|9|3) == 2).all());
        REQUIRE((A.select(_|2|9|3) == 3).all());
        REQUIRE_THROWS_AS(A.select(_|0|9|3)(3), std::out_of_range);
        REQUIRE_THROWS_AS(A.select(_|1|9|3)(3), std::out_of_range);
        REQUIRE_THROWS_AS(A.select(_|2|9|3)(3), std::out_of_range);
    }

    SECTION("skipped assignment on axis 0 to ndarray<2> works as expected on")
    {
        auto A = nd::ndarray<int, 2>(9, 7);
        A.select(_|0|9|3, _|0|7) = 1;
        A.select(_|1|9|3, _|0|7) = 2;
        A.select(_|2|9|3, _|0|7) = 3;
        REQUIRE(A.select(_|0|9|3, _|0|7).size() == 21);
        REQUIRE(A.select(_|1|9|3, _|0|7).size() == 21);
        REQUIRE(A.select(_|2|9|3, _|0|7).size() == 21);
        REQUIRE((A.select(_|0|9|3, _|0|7) == 1).all());
        REQUIRE((A.select(_|1|9|3, _|0|7) == 2).all());
        REQUIRE((A.select(_|2|9|3, _|0|7) == 3).all());
        REQUIRE_THROWS_AS(A.select(_|0|9|3, _|0|7)(3, 0), std::out_of_range);
        REQUIRE_THROWS_AS(A.select(_|1|9|3, _|0|7)(3, 0), std::out_of_range);
        REQUIRE_THROWS_AS(A.select(_|2|9|3, _|0|7)(3, 0), std::out_of_range);
    }

    SECTION("skipped assignment on axis 1 to ndarray<2> works as expected on")
    {
        auto A = nd::ndarray<int, 2>(7, 9);
        A.select(_|0|7, _|0|9|3) = 1;
        A.select(_|0|7, _|1|9|3) = 2;
        A.select(_|0|7, _|2|9|3) = 3;
        REQUIRE(A.select(_|0|7, _|0|9|3).size() == 21);
        REQUIRE(A.select(_|0|7, _|1|9|3).size() == 21);
        REQUIRE(A.select(_|0|7, _|2|9|3).size() == 21);
        REQUIRE((A.select(_|0|7, _|0|9|3) == 1).all());
        REQUIRE((A.select(_|0|7, _|1|9|3) == 2).all());
        REQUIRE((A.select(_|0|7, _|2|9|3) == 3).all());
        REQUIRE_THROWS_AS(A.select(_|0|7, _|0|9|3)(0, 3), std::out_of_range);
        REQUIRE_THROWS_AS(A.select(_|0|7, _|1|9|3)(0, 3), std::out_of_range);
        REQUIRE_THROWS_AS(A.select(_|0|7, _|2|9|3)(0, 3), std::out_of_range);
    }
}


#endif // TEST_NDARRAY
