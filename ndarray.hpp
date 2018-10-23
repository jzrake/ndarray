#include <memory>
#include <vector>
#include <array>
#include <numeric>




// ============================================================================
template<int Rank, int Axis> struct selector;
template<typename Op, int Rank> class binary_operation;
template<int Rank> class ndarray;




// ============================================================================
template<int Rank, int Axis=0>
struct selector
{


    enum { rank = Rank, axis = Axis };


    /**
        Collapse this selector at the given index, creating a selector with
        rank reduced by 1, and which operates on the same axis.
     */
    selector<rank - 1, axis> collapse(int start_index) const
    {
        static_assert(rank > 0, "selector: cannot collapse zero-rank selector");
        static_assert(axis < rank, "selector: attempting to index on axis >= rank");

        std::array<int, rank - 1> _count;
        std::array<int, rank - 1> _start;
        std::array<int, rank - 1> _final;

        for (int n = 0; n < axis; ++n)
        {
            _count[n] = count[n];
            _start[n] = start[n];
            _final[n] = final[n];
        }

        for (int n = axis + 1; n < rank - 1; ++n)
        {
            _count[n] = count[n + 1];
            _start[n] = start[n + 1];
            _final[n] = final[n + 1];
        }

        _count[axis] = count[axis] * count[axis + 1];
        _start[axis] = count[axis] * start[axis] + start[axis + 1] + start_index;
        _final[axis] = count[axis] * start[axis] + final[axis + 1];

        return {_count, _start, _final};
    }

    /**
        Alias for the collapse function.
     */
    selector<rank - 1, axis> select(int start_index) const
    {
        return collapse(start_index);
    }

    /**
        Return a subset of this selector by specifying a range on each axis.
     */
    selector<rank, axis + 1> select(std::tuple<int, int> range) const
    {
        static_assert(axis < rank, "selector: attempting to index on axis >= rank");

        auto _count = count;
        auto _start = start;
        auto _final = final;

        _start[axis] = start[axis] + std::get<0>(range);
        _final[axis] = start[axis] + std::get<1>(range);

        return {_count, _start, _final};
    }

    /**
        Return a selector of same or smaller rank, applying the collapse or select
        operators sequentially to each axis.
     */
    template<typename First, typename... Rest>
    auto select(First first, Rest... rest) const
    {
        return select(first).select(rest...);
    }

    /**
        Return a selector covering the same sub-space but operating on axis 0.
     */
    selector<rank> reset() const
    {
        return selector<rank>{count, start, final};
    }

    /**
        Return the shape of the sub-space covered by this selector.
     */
    std::array<int, rank> shape() const
    {
        std::array<int, rank> s;

        for (int n = 0; n < rank; ++n)
        {
            s[n] = final[n] - start[n];
        }
        return s;
    }

    /**
        Return the number of elements in the sub-space covered by this selector.
     */
    int size() const
    {
        auto s = shape();
        return std::accumulate(s.begin(), s.end(), 1, std::multiplies<>());
    }

    bool operator==(const selector<rank, axis>& other) const
    {
        return count == other.count && start == other.start && final == other.final;
    }

    bool operator!=(const selector<rank, axis>& other) const
    {
        return ! operator==(other);
    }

    bool next(std::array<int, rank>& index) const
    {
        int n = rank - 1;

        ++index[n];

        while (index[n] == final[n])
        {
            if (n == 0)
            {
                index = final;
                return false;
            }
            index[n] = start[n];
            ++index[--n];
        }
        return true;
    }

    /**
        Return another selector for the same rank, axis, and shape, but with
        start = 0 and count = shape.
     */
    selector<rank, axis> normalize() const
    {
        auto origin = start;
        for (auto& si : origin) si = 0;
        return {count, origin, count};
    }




    // ========================================================================
    class iterator
    {
    public:
        iterator(selector<rank> sel, std::array<int, rank> ind) : sel(sel), ind(ind) {}
        iterator& operator++() { sel.next(ind); return *this; }
        iterator operator++(int) { auto ret = *this; this->operator++(); return ret; }
        bool operator==(iterator other) const { return ind == other.ind; }
        bool operator!=(iterator other) const { return ind != other.ind; }
        const std::array<int, rank>& operator*() const { return ind; }
    private:
        std::array<int, rank> ind;
        selector<rank> sel;
    };

    iterator begin() const { return {reset(), start}; }
    iterator end() const { return {reset(), final}; }




    /** Data members
     *
     */
    // ========================================================================
    std::array<int, rank> count;
    std::array<int, rank> start;
    std::array<int, rank> final;
};




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
    ndarray(const ndarray<rank>& other)
    {
        *this = other.copy();
    }

    ndarray(ndarray<rank>& other)
    {
        *this = other;
    }

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
    , skips(constant_array<int, rank>(1))
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
        assert(data->size() == std::accumulate(count.begin(), count.end(), 1, std::multiplies<>()));
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
        assert(data->size() == std::accumulate(count.begin(), count.end(), 1, std::multiplies<>()));
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
        return {make_selector().collapse(index), data};
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
        return selector<rank>{count, start, final};
    }

    template <int R = rank, typename std::enable_if_t<R == 0>* = nullptr>
    static std::array<int, rank> compute_strides(std::array<int, rank> count)
    {
        return std::array<int, rank>();
    }

    template <int R = rank, typename std::enable_if_t<R != 0>* = nullptr>
    static std::array<int, rank> compute_strides(std::array<int, rank> count)
    {
        std::array<int, rank> s;   
        std::partial_sum(count.rbegin(), count.rend() - 1, s.rbegin() + 1, std::multiplies<>());
        s[rank - 1] = 1;
        return s;
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
