#include <memory>
#include <vector>
#include <array>
#include <numeric>




// ============================================================================
template <int rank>
class ndarray
{
public:

    template <typename... Dims>
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

    ndarray<rank - 1> operator[](int index)
    {
        auto s = strides();

        std::array<int, rank - 1> _count;
        std::array<int, rank - 1> _start;
        std::array<int, rank - 1> _stop;
        std::array<int, rank - 1> _skip;

        _count[0] = count[0] * count[1];
        _start[0] = start[0] * s[0] + skip[0] * index * s[0];
        _stop[0] = _start[0] + count[1];
        _skip[0] = skip[1];

        for (int n = 1; n < rank - 1; ++n)
        {
            _count[n] = count[n + 1];
            _start[n] = start[n + 1];
            _stop[n] = stop[n + 1];
            _skip[n] = skip[n + 1];
        }

        return ndarray<rank - 1>(_count, _start, _stop, _skip, data);
    }

    template<int axis>
    ndarray<rank> select(int start_index, int stop_index) const
    {
        auto _count = count;
        auto _start = start;
        auto _stop = stop;
        auto _skip = skip;

        _start[axis] = _start[axis] + start_index;
        _stop[axis] = _start[axis] + stop_index;

        return ndarray<rank> (_count, _start, _stop, _skip, data);
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
        std::array<int, rank> s;
        s[rank - 1] = 1;

        for (int n = rank - 2; n >= 0; --n)
        {
            s[n] = s[n + 1] * count[n + 1];
        }
        return s;
    }

    template <typename... Index>
    size_t offset(Index... index) const
    {
        static_assert(sizeof...(index) == rank,
          "Number of arguments to ndarray::offset must match rank");

        return offset({index...});
    }

    template <typename... Index>
    double& operator()(Index... index)
    {
        static_assert(sizeof...(index) == rank,
          "Number of arguments to ndarray::operator() must match rank");

        return data->operator[](offset({index...}));
    }

    template <typename... Index>
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
