#pragma once
#include <array>
#include <numeric>
#include <string>
#include <memory>
#include <cstring>
#include <functional>




// ============================================================================
namespace nd 
{
    template<int Rank, int Axis> struct selector;

    /**
     * Creates a selector without count (memory extent) information, e.g:
     *
     * auto sel = make_selector(_0|100|2, _|50|100);
     *
     * This selector can return size, skips, and shape information, but thinks
     * it has a count of 1 on each axis.
     */
    template<typename... Index>
    static inline auto make_selector(Index... index)
    {
        return selector<sizeof...(Index), 0>().select(index...);
    }

    /**
     * Returns a selector with a different count, by reading from the given
     * iterator range. The iterator difference must match the selector rank,
     * but no checking is performed that the existing selection is within the
     * new count.
     */
    template<typename Selector, typename First, typename Second>
    static inline auto with_count(Selector sel, First begin, Second end)
    {
        auto it = begin;
        auto n = 0;

        if (end - begin != sel.rank)
        {
            throw std::invalid_argument("with_count got wrong number of axes");
        }
        while (it != end)
        {
            sel.count[n++] = *it++;
        }
        return sel;
    }
} 




// ============================================================================
namespace nd 
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
        inline std::array<std::tuple<int, int>, rank> promote(std::array<std::tuple<int, int>, rank> shape);
        inline std::array<std::tuple<int, int>, 1> promote(std::tuple<int, int, int> selection);
        inline std::array<std::tuple<int, int>, 1> promote(std::tuple<int, int> range);
        inline std::array<std::tuple<int, int>, 1> promote(int index);
        inline std::array<std::tuple<int, int>, 1> promote(axis::selection selection);
        inline std::array<std::tuple<int, int>, 1> promote(axis::range range);
        inline std::array<std::tuple<int, int>, 1> promote(axis::index index);
        inline std::array<std::tuple<int, int>, 1> promote(axis::all all);
        template<typename First>                   inline auto make_shape(First first);
        template<typename First, typename Second>  inline auto make_shape(First first, Second second);
        template<typename First, typename... Rest> inline auto make_shape(First first, Rest... rest);


        /**
         * Helper function to convert shapes to string. Very handy in error
         * reporting.
         */
        template<typename T, std::size_t Size>
        std::string static inline to_string(std::array<T, Size> a)
        {
            auto res = std::string("[");

            for (std::size_t n = 0; n < Size; ++n)
            {
                res += std::to_string(a[n]) + (n == Size - 1 ? "" : " ");
            }
            return res + "]";
        }
    }
} 




// ============================================================================
namespace nd 
{
    template<typename T> class buffer;
} 




// ============================================================================
namespace nd 
{
    template<typename T, typename U, int R, typename Op> struct binary_op;
    template<typename T, int R, typename Op> struct unary_op;
    template<typename T, int R> class ndarray;
    template<typename T> struct dtype_str;

    template<typename T> ndarray<T, 1> static inline arange(int size);
    template<typename T> ndarray<T, 1> static inline linspace(T start, T end, int size);
    template<typename T> ndarray<T, 1> static inline ones(int size);
    template<typename T> ndarray<T, 1> static inline zeros(int size);

    template<typename T, int R>
    static inline nd::ndarray<T, R + 1> stack(std::initializer_list<nd::ndarray<T, R - 1>> arrays);

/**
 * Unless you define the following macro, an alias nd::array will be created
 * for you, to make your declarations a little cleaner.
 */
#ifndef ND_DONT_ALIAS_TO_ARRAY
    template<typename T, int R> using array = ndarray<T, R>;
#endif
} 




// ============================================================================
template<int Rank, int Axis = 0> 
struct nd::selector
{


    enum { rank = Rank, axis = Axis };


    // ========================================================================
    selector()
    {
        for (int n = 0; n < rank; ++n)
        {
            count[n] = 1;
            start[n] = 0;
            final[n] = 1;
            skips[n] = 1;
        }
    }

    template<typename... Dims>
    selector(Dims... dims) : selector(std::array<int, rank>{dims...})
    {
        static_assert(sizeof...(Dims) == rank,
            "selector: number of count arguments must match rank");
    }

    selector(std::array<int, rank> count) : count(count)
    {
        for (int n = 0; n < rank; ++n)
        {
            start[n] = 0;
            final[n] = count[n];
            skips[n] = 1;
        }
    }

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
  template <int R = rank, int A = axis, typename std::enable_if<A == rank - 1>::type* = nullptr>
    selector<rank - 1, axis - 1> collapse() const
    {
        static_assert(rank > 0, "selector: cannot collapse zero-rank selector");

        std::array<int, rank - 1> _count;
        std::array<int, rank - 1> _start;
        std::array<int, rank - 1> _final;
        std::array<int, rank - 1> _skips;

        for (int n = 0; n < rank - 2; ++n)
        {
            _count[n] = count[n];
            _start[n] = start[n];
            _final[n] = final[n];
            _skips[n] = skips[n];
        }

        _count[axis - 1] = count[axis] * count[axis - 1];
        _start[axis - 1] = count[axis] * start[axis - 1] + start[axis];
        _final[axis - 1] = count[axis] * final[axis - 1] + start[axis];
        _skips[axis - 1] = count[axis];

        return {_count, _start, _final, _skips};
    }

    template <int R = rank, int A = axis, typename std::enable_if<A < rank - 1>::type* = nullptr>
    selector<rank - 1, axis> collapse() const
    {
        static_assert(rank > 0, "selector: cannot collapse zero-rank selector");

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

        _count[axis] = count[axis + 1] * count[axis];
        _start[axis] = count[axis + 1] * start[axis] + start[axis + 1];
        _final[axis] = count[axis + 1] * final[axis] + start[axis + 1];
        _skips[axis] = 1;

        return {_count, _start, _final, _skips};
    }

    selector<rank, axis + 1> skip(int skips_index) const
    {
        return select(std::make_tuple(start[axis], final[axis], skips_index));
    }

    selector<rank, axis + 1> slice(int lower_index, int upper_index, int skips_index) const
    {
        static_assert(axis < rank, "selector: cannot select on axis >= rank");

        auto _count = count;
        auto _start = start;
        auto _final = final;
        auto _skips = skips;

        _start[axis] = start[axis] + lower_index;
        _final[axis] = start[axis] + upper_index;
        _skips[axis] = skips[axis] * skips_index;

        return {_count, _start, _final, _skips};
    }

    selector<rank, axis + 1> select(axis::selection selection) const
    {
        return slice(selection.lower, selection.upper, selection.skips);
    }

    selector<rank, axis + 1> select(axis::range range) const
    {
        return slice(range.lower, range.upper, 1);
    }

    auto select(axis::index index) const
    {
        return slice(index.lower);
    }

    selector<rank, axis + 1> select(axis::all) const
    {
        return {count, start, final, skips};
    }

    selector<rank, axis + 1> select(std::tuple<int, int, int> selection) const
    {
        return slice(
            std::get<0>(selection),
            std::get<1>(selection),
            std::get<2>(selection));
    }

    selector<rank, axis + 1> select(std::tuple<int, int> range) const
    {
        return slice(std::get<0>(range), std::get<1>(range), 1);
    }

    auto select(int index) const
    {
        return slice(index, index + 1, 1).drop().collapse();
    }

    template<typename First, typename... Rest>
    auto select(First first, Rest... rest) const
    {
        return select(first).select(rest...);
    }

    template<int new_axis>
    selector<rank, new_axis> on() const
    {
        static_assert(new_axis >= 0 && new_axis < rank, "invalid selector axis");
        return {count, start, final, skips};
    }

    selector<rank> reset() const
    {
        return {count, start, final, skips};
    }

    selector<rank, axis - 1> drop() const
    {
        static_assert(axis > 0, "invalid selector axis");
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
        return s;
    }

    std::array<int, rank> shape() const
    {
        std::array<int, rank> s;

        for (int n = 0; n < rank; ++n)
        {
            s[n] = shape(n);
        }
        return s;
    }

    int shape(int axis) const
    {
        return final[axis] / skips[axis] - start[axis] / skips[axis];
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

    std::size_t size() const
    {
        auto s = shape();
        return std::accumulate(s.begin(), s.end(), 1, std::multiplies<int>());
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

        while (index[n] >= final[n])
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

    template<typename... Index>
    bool contains(Index... index) const
    {
        static_assert(sizeof...(Index) == rank, "selector: index size must match rank");

        auto S = shape::make_shape(index...);

        for (int n = 0; n < rank; ++n)
        {
            auto start_index = std::get<0>(S[n]);
            auto final_index = std::get<1>(S[n]);

            if (start_index < 0 || final_index > final[n] / skips[n] - start[n] / skips[n])
            {
                return false;
            }
        }
        return true;
    }

    selector<rank, axis> shift(int dist) const
    {
        auto sel = *this;
        sel.start[axis] = std::max(sel.start[axis] + dist * skips[axis], 0);
        sel.final[axis] = std::min(sel.final[axis] + dist * skips[axis], sel.count[axis]);
        return sel;
    }




    // ========================================================================
    class iterator
    {
    public:
        iterator() {}
        iterator(selector<rank> sel, std::array<int, rank> ind) : sel(sel), ind(ind) {}
        iterator& operator++() { sel.next(ind); return *this; }
        iterator operator++(int) { auto ret = *this; this->operator++(); return ret; }
        bool operator==(iterator other) const { return ind == other.ind; }
        bool operator!=(iterator other) const { return ind != other.ind; }
        const std::array<int, rank>& operator*() const { return ind; }
    private:
        selector<rank> sel;
        std::array<int, rank> ind;
    };

    iterator begin() const { return {reset(), start}; }
    iterator end() const { return {reset(), final}; }




    // ========================================================================
    std::array<int, rank> count;
    std::array<int, rank> start;
    std::array<int, rank> final;
    std::array<int, rank> skips;




    // ========================================================================
    template<int other_rank, int other_axis>
    friend struct selector;
}; 




// ============================================================================
template<unsigned long rank> 
std::array<std::tuple<int, int>, rank> nd::shape::promote(std::array<std::tuple<int, int>, rank> shape)
{
    return shape;
}

std::array<std::tuple<int, int>, 1> nd::shape::promote(std::tuple<int, int, int> selection)
{
    return {std::make_tuple(std::get<0>(selection), std::get<1>(selection))};
}

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
} 




// ============================================================================
template<typename T> 
class nd::buffer
{
public:

    buffer() {}

    buffer(const buffer<T>& other)
    {
        count = other.count;
        memory = new T[count];

        for (int n = 0; n < count; ++n)
        {
            memory[n] = other.memory[n];
        }
    }

    buffer(buffer<T>&& other)
    {
        delete [] memory;
        memory = other.memory;
        count = other.count;

        other.memory = nullptr;
        other.count = 0;
    }

    explicit buffer(std::size_t count, const T& value = T()) : count(count)
    {
        memory = new T[count];

        for (std::size_t n = 0; n < count; ++n)
        {
            memory[n] = value;
        }
    }

    template< class InputIt >
    buffer(InputIt first, InputIt last)
    {
        {
            auto it = first;
            count = 0;

            while (it != last)
            {
                ++it;
                ++count;
            }
            memory = new T[count];
        }

        {
            auto it = first;
            auto n = 0;

            while (it != last)
            {
                memory[n] = *it;
                ++it;
                ++n;
            }
        }
    }

    ~buffer()
    {
        delete [] memory;
    }

    buffer<T>& operator=(const buffer<T>& other)
    {
        delete [] memory;

        count = other.count;
        memory = new T[count];

        for (int n = 0; n < count; ++n)
        {
            memory[n] = other.memory[n];
        }
        return *this;
    }

    buffer<T>& operator=(buffer<T>&& other)
    {
        delete [] memory;
        memory = other.memory;
        count = other.count;

        other.memory = nullptr;
        other.count = 0;
        return *this;
    }

    bool operator==(const buffer<T>& other) const
    {
        if (count != other.count)
        {
            return false;
        }

        for (std::size_t n = 0; n < size(); ++n)
        {
            if (memory[n] != other.memory[n])            
            {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const buffer<T>& other) const
    {
        return ! operator==(other);
    }

    std::size_t size() const
    {
        return count;
    }

    const T* data() const
    {
        return memory;
    }

    T* data()
    {
        return memory;
    }

    const T& operator[](std::size_t offset) const
    {
        return memory[offset];
    }

    T& operator[](std::size_t offset)
    {
        return memory[offset];
    }

    T* begin() { return memory; }
    T* end() { return memory + count; }

    const T* begin() const { return memory; }
    const T* end() const { return memory + count; }

private:
    T* memory = nullptr;
    std::size_t count = 0;
}; 




// ============================================================================
template<typename T> nd::ndarray<T, 1> nd::arange(int size) 
{
    auto A = nd::ndarray<T, 1>(size);
    auto x = T();
    for (auto& a : A) a = x++;
    return A;
}

template<typename T> nd::ndarray<T, 1> nd::linspace(T start, T end, int size)
{
    auto A = nd::ndarray<T, 1>(size);
    auto h = (end - start) / (size - 1);
    auto x = start - h;
    for (auto& a : A) a = (x += h);
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
/**
 * This block can be expanded to accommodate new data types. Note: gcc requires
 * these template specializations to be declared directly inside the namespace.
 */
namespace nd {
  template<> struct dtype_str<bool  > { static inline std::array<char, 8> value(); };
  template<> struct dtype_str<float > { static inline std::array<char, 8> value(); };
  template<> struct dtype_str<double> { static inline std::array<char, 8> value(); };
  template<> struct dtype_str<int   > { static inline std::array<char, 8> value(); };
  template<> struct dtype_str<long  > { static inline std::array<char, 8> value(); };

  std::array<char, 8> dtype_str<bool  >::value() { return {'b','1',  0,  0,  0,  0,  0,  0}; }
  std::array<char, 8> dtype_str<float >::value() { return {'f','4',  0,  0,  0,  0,  0,  0}; }
  std::array<char, 8> dtype_str<double>::value() { return {'f','8',  0,  0,  0,  0,  0,  0}; }
  std::array<char, 8> dtype_str<int   >::value() { return {'i','4',  0,  0,  0,  0,  0,  0}; }
  std::array<char, 8> dtype_str<long  >::value() { return {'i','8',  0,  0,  0,  0,  0,  0}; }
}




// ============================================================================
template<typename T, int R>
class nd::ndarray
{
public:


    using dtype = T;
    enum { rank = R };




    /**
     * Wrapper class for constant arrays. Can be trivially converted back to
     * const nd::array&.
     */
    // ========================================================================
    class const_ref
    {
    public:

        using dtype = T;
        enum { rank = R };

        const_ref(selector<R> sel, std::shared_ptr<buffer<T>> buf) : A(sel, buf) {}
        template<typename... Args> auto operator[](Args... args) const { return A.operator[](args...); }
        template<typename... Args> auto operator()(Args... args) const { return A.operator()(args...); }
        template<typename... Args> auto shape(const Args&... args) const { return A.shape(args...); }
        template<typename... Args> auto size(const Args&... args) const { return A.size(args...); }
        template<typename... Args> auto shares(const Args&... args) const { return A.shares(args...); }
        template<int Axis, typename... Args> auto take(const Args&... args) const { return A.take<Axis>(args...); }
        template<int Axis, typename... Args> auto shift(const Args&... args) const { return A.shift<Axis>(args...); }

        operator const ndarray<T, R>&() const { return A; }
        bool is_const_ref() const { return true; }

        auto begin() const { return A.begin(); }
        auto end() const { return A.end(); }

    private:
        friend class ndarray;
        ndarray<T, R> A;
    };




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
    , strides(sel.strides())
    , buf(std::make_shared<buffer<T>>(elements.begin(), elements.end()))
    {
    }

    template<typename... Dims>
    ndarray(Dims... dims) : ndarray(std::array<int, R>({int(dims)...}))
    {
        static_assert(sizeof...(dims) == rank,
          "Number of arguments to ndarray constructor must match rank");
    }

    template<typename SelectorType>
    ndarray(SelectorType sel, std::shared_ptr<buffer<T>>& buf)
    : sel(sel)
    , strides(sel.strides())
    , buf(buf)
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
    , strides(sel.strides())
    , buf(buf)
    {
        assert_valid_argument(buf->size() == sel.size(),
            "Size of data buffer is not the product of dim sizes");
    }

    ndarray(const ndarray<T, R>& other)
    : sel(other.sel.shape())
    , strides(sel.strides())
    , buf(std::make_shared<buffer<T>>(size()))
    {
        copy_internal(*this, other);
    }

    ndarray(ndarray<T, R>& other)
    {
        strides = other.strides;
        sel = other.sel;
        buf = other.buf;
    }




    /**
     * Assignment operators
     * 
     */
    // ========================================================================
    template <int Rank = R, typename std::enable_if<Rank == 0>::type* = nullptr>
    ndarray<T, R>& operator=(T value)
    {
        buf->operator[](scalar_offset) = value;
        return *this;
    }

    template <int Rank = R, typename std::enable_if<Rank != 0>::type* = nullptr>
    ndarray<T, R>& operator=(T value)
    {
        for (auto& a : *this)
            a = value;

        return *this;
    }

    ndarray<T, R>& operator=(const ndarray<T, R>& other)
    {
        copy_internal(*this, other);
        return *this;
    }

    void become(ndarray<T, R> other)
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
    const ndarray<T, sizeof...(Sizes)> reshape(Sizes... sizes) const
    {
        if (! contiguous())
        {
            auto A = ndarray<T, sizeof...(Sizes)>(sizes...);
            copy_internal(A, *this);
            return A;
        }
        return ndarray<T, sizeof...(Sizes)>({sizes...}, const_cast<std::shared_ptr<buffer<T>>&>(buf));
    }




    /**
     * Shape and size query methods
     * 
     */
    // ========================================================================
    auto size() const { return sel.size(); }
    bool empty() const { return sel.empty(); }
    auto shape() const { return sel.shape(); }
    auto shape(int axis) const { return sel.shape(axis); }
    bool contiguous() const { return sel.contiguous(); }




    /**
     * Data accessors and selection methods
     * 
     */
    // ========================================================================
    template <int Rank = R, typename std::enable_if<Rank == 1>::type* = nullptr>
    ndarray<T, R - 1> operator[](int index)
    {
        if (index < 0 || index >= (sel.final[0] - sel.start[0]) / sel.skips[0])
            throw std::out_of_range("ndarray: index out of range");

        return {offset_relative({index}), buf};
    }

    template <int Rank = R, typename std::enable_if<Rank == 1>::type* = nullptr>
    const ndarray<T, R - 1> operator[](int index) const
    {
        if (index < 0 || index >= (sel.final[0] - sel.start[0]) / sel.skips[0])
            throw std::out_of_range("ndarray: index out of range");

        return {offset_relative({index}), const_cast<std::shared_ptr<buffer<T>>&>(buf)};
    }

    template <int Rank = R, typename std::enable_if<Rank != 1>::type* = nullptr>
    ndarray<T, R - 1> operator[](int index)
    {
        if (index < 0 || index >= (sel.final[0] - sel.start[0]) / sel.skips[0])
            throw std::out_of_range("ndarray: index out of range");

        return {sel.select(index), buf};
    }

    template <int Rank = R, typename std::enable_if<Rank != 1>::type* = nullptr>
    const ndarray<T, R - 1> operator[](int index) const
    {
        if (index < 0 || index >= (sel.final[0] - sel.start[0]) / sel.skips[0])
            throw std::out_of_range("ndarray: index out of range");

        return {sel.select(index), const_cast<std::shared_ptr<buffer<T>>&>(buf)};
    }

    template<typename... Index>
    T& operator()(Index... index)
    {
        if (! sel.contains(index...))
            throw std::out_of_range("ndarray: index out of range");

        return buf->operator[](offset_relative({int(index)...}));
    }

    template<typename... Index>
    const T& operator()(Index... index) const
    {
        if (! sel.contains(index...))
            throw std::out_of_range("ndarray: selection out of range");

        return buf->operator[](offset_relative({int(index)...}));
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
    auto select(Index... index) const
    {
        if (! sel.contains(index...))
            throw std::out_of_range("ndarray: selection out of range");

        auto S = sel.select(index...);
        return typename ndarray<T, S.rank>::const_ref(S.reset(), buf);
    }

    template<int Axis, typename Slice>
    auto take(Slice slice)
    {
        auto taken_sel = sel.template on<Axis>().select(slice).reset();
        return ndarray<T, R>(taken_sel, buf);
    }

    template<int Axis, typename Slice>
    auto take(Slice slice) const
    {
        auto S = sel.template on<Axis>().select(slice).reset();
        return typename ndarray<T, S.rank>::const_ref(S.reset(), buf);
    }

    template<int Axis>
    auto shift(int distance)
    {
        auto shifted_sel = sel.template on<Axis>().shift(distance).reset();
        return ndarray<T, R>(shifted_sel, buf);
    }

    template<int Axis>
    auto shift(int distance) const
    {
        auto S = sel.template on<Axis>().shift(distance).reset();
        return typename ndarray<T, S.rank>::const_ref(S.reset(), buf);
    }

    template <int Rank = R, typename std::enable_if<Rank == 0>::type* = nullptr>
    operator T() const
    {
        // static_assert(rank == 0, "can only convert rank-0 array to scalar value");
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

    T* data()
    {
        return buf->data();
    }

    selector<R> get_selector() const
    {
        return sel;
    }

    bool is_const_ref() const
    {
        return false;
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

        iterator() {}
        iterator(ndarray<T, R>& array, typename selector<rank>::iterator it)
        : mem(array.buf->data())
        , strides(array.strides)
        , it(it)
        {
        }

        iterator& operator++() { it.operator++(); return *this; }
        iterator operator++(int) { auto ret = *this; this->operator++(); return ret; }
        bool operator==(iterator other) const { return mem == other.mem && it == other.it; }
        bool operator!=(iterator other) const { return mem != other.mem || it != other.it; }
        T& operator*() { return mem[offset_absolute(*it)]; }

    private:
        int offset_absolute(std::array<int, R> index) const
        {
            int m = 0;
            for (int n = 0; n < rank; ++n) m += index[n] * strides[n];
            return m;
        }
        T* mem = nullptr;
        std::array<int, R> strides = ndarray::constant_array<R>(0);
        typename selector<rank>::iterator it;
    };

    iterator begin() { static_assert(R > 0, "cannot iterate over scalar"); return {*this, sel.begin()}; }
    iterator end()   { static_assert(R > 0, "cannot iterate over scalar"); return {*this, sel.end()}; }




    // ========================================================================
    class const_iterator
    {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = const T*;
        using reference = const T&;
        using iterator_category = std::forward_iterator_tag;

        const_iterator() {}
        const_iterator(const ndarray<T, R>& array, typename selector<rank>::iterator it)
        : mem(array.buf->data())
        , strides(array.strides)
        , it(it)
        {
        }

        const_iterator& operator++() { it.operator++(); return *this; }
        const_iterator operator++(int) { auto ret = *this; this->operator++(); return ret; }
        bool operator==(const_iterator other) const { return mem == other.mem && it == other.it; }
        bool operator!=(const_iterator other) const { return mem != other.mem || it != other.it; }
        const T& operator*() { return mem[offset_absolute(*it)]; }

    private:
        int offset_absolute(std::array<int, R> index) const
        {
            int m = 0;
            for (int n = 0; n < rank; ++n) m += index[n] * strides[n];
            return m;
        }
        const T* mem = nullptr;
        std::array<int, R> strides = ndarray::constant_array<R>(0);
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

        auto D = dtype_str<T>::value();
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

        assert_valid_argument(it + sizeof(D) <= str.end(), "unexpected end of ndarray header string");
        std::memcpy(&D, &*it, sizeof(D));
        it += sizeof(D);

        assert_valid_argument(it + sizeof(Q) <= str.end(), "unexpected end of ndarray header string");
        std::memcpy(&Q, &*it, sizeof(Q));
        it += sizeof(Q);

        assert_valid_argument(it + sizeof(S) <= str.end(), "unexpected end of ndarray header string");
        std::memcpy(&S, &*it, sizeof(S));
        it += sizeof(S);

        assert_valid_argument(D == dtype_str<T>::value(), "ndarray string has wrong data type");
        assert_valid_argument(Q == rank, "ndarray string has the wrong rank");

        auto size = std::accumulate(S.begin(), S.end(), 1, std::multiplies<int>());
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
        {
            throw std::invalid_argument(message);
        }
    }

    template<typename TT, int TR>
    static void copy_internal(ndarray<TT, TR>& target, const ndarray<T, R>& source)
    {
        if (target.size() != source.size())
        {
            throw std::invalid_argument("incompatible assignment from "
                + shape::to_string(source.shape())
                + " to "
                + shape::to_string(target.shape()));
        }

        auto a = target.begin();
        auto b = source.begin();

        for (; a != target.end(); ++a, ++b)
        {
            *a = *b;
        }
    }

    static void copy_internal(ndarray<T, R>& target, const ndarray<T, R>& source)
    {
        if (target.shape() != source.shape())
        {
            throw std::invalid_argument("incompatible assignment from "
                + shape::to_string(source.shape())
                + " to "
                + shape::to_string(target.shape()));
        }

        auto a = target.begin();
        auto b = source.begin();

        for (; a != target.end(); ++a, ++b)
        {
            *a = *b;
        }
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
}; 
