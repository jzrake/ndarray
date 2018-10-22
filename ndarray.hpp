#include <memory>
#include <vector>
#include <array>
#include <numeric>




// ============================================================================
template<int rank>
class ndarray
{
public:




    // ========================================================================
    template<typename S, int a>
    struct recurse_selector
    {
        static ndarray<rank> select(ndarray<rank> A, S s)
        {
            return recurse_selector<S, a - 1>::select(A.select(a, std::get<0>(s[a]), std::get<1>(s[a])), s);
        }
    };

    template<typename S>
    struct recurse_selector<S, 0>
    {
        static ndarray<rank> select(ndarray<rank> A, S s)
        {
            return A.select(0, std::get<0>(s[0]), std::get<1>(s[0]));
        }
    };




    // ========================================================================
    template<typename... Dims>
    ndarray(Dims... dims) : ndarray(std::array<int, rank>({dims...}))
    {
        static_assert(sizeof...(dims) == rank,
          "Number of arguments to ndarray constructor must match rank");
    }

    ndarray(std::array<int, rank> dim_sizes)
    : count(dim_sizes)
    , start(constant_int_array<rank>(0))
    , stop(dim_sizes)
    , skip(constant_int_array<rank>(1))
    , data(std::make_shared<std::vector<double>>(product(dim_sizes)))
    {}

    ndarray(std::array<int, rank> dim_sizes, std::shared_ptr<std::vector<double>> data)
    : count(dim_sizes)
    , start(constant_int_array<rank>(0))
    , stop(dim_sizes)
    , skip(constant_int_array<rank>(1))
    , data(data)
    {
        assert(data->size() == std::accumulate(count.begin(), count.end(), 1, std::multiplies<>()));
    }

    ndarray(
        std::array<int, rank> count,
        std::array<int, rank> start,
        std::array<int, rank> stop,
        std::array<int, rank> skip,
        std::shared_ptr<std::vector<double>> data)
    : count(count)
    , start(start)
    , stop(stop)
    , skip(skip)
    , data(data)
    {
        assert(data->size() == std::accumulate(count.begin(), count.end(), 1, std::multiplies<>()));
    }




    // ========================================================================
    ndarray<rank - 1> operator[](int index)
    {
        return collapse(0, index);
    }

    ndarray<rank - 1> collapse(int axis, int start_index) const
    {
        auto s = strides();

        std::array<int, rank - 1> _count;
        std::array<int, rank - 1> _start;
        std::array<int, rank - 1> _stop;
        std::array<int, rank - 1> _skip;

        for (int n = 0; n < axis; ++n)
        {
            _count[n] = count[n];
            _start[n] = start[n];
            _stop[n] = stop[n];
            _skip[n] = skip[n];
        }

        for (int n = axis + 1; n < rank - 1; ++n)
        {
            _count[n] = count[n + 1];
            _start[n] = start[n + 1];
            _stop[n] = stop[n + 1];
            _skip[n] = skip[n + 1];
        }

        _count[axis] = count[axis] * count[axis + 1];
        _start[axis] = start[axis] * s[axis] + skip[axis] * start_index * s[axis];
        _stop[axis] = _start[axis] + count[axis + 1];
        _skip[axis] = skip[axis + 1];

        return ndarray<rank - 1>(_count, _start, _stop, _skip, data);
    }

    ndarray<rank> select(int axis, int start_index, int stop_index) const
    {
        auto _count = count;
        auto _start = start;
        auto _stop = stop;
        auto _skip = skip;

        _start[axis] = start[axis] + start_index;
        _stop[axis] = start[axis] + stop_index;

        return ndarray<rank>(_count, _start, _stop, _skip, data);
    }

    template<typename... Slices>
    ndarray<rank> within(Slices... slices) const
    {
        static_assert(sizeof...(slices) == rank,
          "Number of arguments to ndarray::within must match rank");

        return recurse_selector<
        std::array<std::tuple<int, int>, rank>,
        rank - 1>::select(*this, {slices...});
    }

    size_t size() const
    {
        auto s = shape();
        return std::accumulate(s.begin(), s.end(), 1, std::multiplies<>());
    }

    std::array<int, rank> shape() const
    {
        std::array<int, rank> s;

        for (int n = 0; n < rank; ++n)
        {
            s[n] = (stop[n] - start[n]) / skip[n];
        }
        return s;
    }

    std::array<int, rank> strides() const
    {
        std::array<int, rank> s = {1};
        std::partial_sum(count.rbegin(), count.rend() - 1, s.rbegin() + 1, std::multiplies<int>());
        s[rank - 1] = 1;
        return s;
    }

    template<typename... Index>
    size_t offset(Index... index) const
    {
        static_assert(sizeof...(index) == rank,
          "Number of arguments to ndarray::offset must match rank");

        return offset({index...});
    }

    template<typename... Index>
    double& operator()(Index... index)
    {
        static_assert(sizeof...(index) == rank,
          "Number of arguments to ndarray::operator() must match rank");

        return data->operator[](offset({index...}));
    }

    template<typename... Index>
    const double& operator()(Index... index) const
    {
        static_assert(sizeof...(index) == rank,
          "Number of arguments to ndarray::operator() must match rank");

        return data->operator[](offset({index...}));
    }

    template<int new_rank>
    ndarray<new_rank> reshape(std::array<int, new_rank> dim_sizes) const
    {
        assert(contiguous());
        return ndarray<new_rank>(dim_sizes, data);
    }

    template<typename... DimSizes>
    ndarray<sizeof...(DimSizes)> reshape(DimSizes... dim_sizes) const
    {
        assert(contiguous());
        return ndarray<sizeof...(DimSizes)>({dim_sizes...}, data);
    }

    bool contiguous() const
    {
        for (int n = 0; n < rank; ++n)
        {
            if (start[n] != 0 || stop[n] != count[n] || skip[n] != 1)
            {
                return false;
            }
        }
        return true;
    }

    template<int other_rank>
    bool shares(const ndarray<other_rank>& other) const
    {
        return data == other.data;
    }

    /**
     * Private methods
     * 
     */
private:
    size_t offset(std::array<int, rank> index) const
    {
        size_t m = 0;
        size_t s = 1;

        for (int n = rank - 1; n >= 0; --n)
        {
            m += start[n] * s + skip[n] * index[n] * s;
            s *= count[n];
        }
        return m;
    }

    template<int length>
    static std::array<int, length> constant_int_array(int value)
    {
        std::array<int, length> A;

        for (auto& a : A)
        {
            a = value;
        }
        return A;
    }

    template<typename C>
    static size_t product(const C& c)
    {
        return std::accumulate(c.begin(), c.end(), 1, std::multiplies<>());
    }

    std::shared_ptr<std::vector<double>> data;
    std::array<int, rank> count;
    std::array<int, rank> start;
    std::array<int, rank> stop;
    std::array<int, rank> skip;

    /** Grants friendship to ndarrays of all other ranks.
     */
    template<int other_rank>
    friend class ndarray;
};
