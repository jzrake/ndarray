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
    {
        size_t total_size = 1;

        for (auto dim = 0; dim < rank; ++dim)
        {
            total_size *= dim_sizes[dim];

            count[dim] = dim_sizes[dim];
            start[dim] = 0;
            stop[dim] = dim_sizes[dim];
            skip[dim] = 1;
        }
        data = std::make_shared<std::vector<double>>(total_size);
    }

    ndarray(
        std::array<size_t, rank> count,
        std::array<size_t, rank> start,
        std::array<size_t, rank> stop,
        std::array<size_t, rank> skip,
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

        std::array<size_t, rank - 1> _count;
        std::array<size_t, rank - 1> _start;
        std::array<size_t, rank - 1> _stop;
        std::array<size_t, rank - 1> _skip;

        // std::cout << "start[0]: " << start[0] << std::endl;
        // std::cout << "skip[0]: " << skip[0] << std::endl;
        // std::cout << "s[0]: " << s[0] << std::endl;

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

        // std::cout << "count[0]: " << _count[0] << std::endl;
        // std::cout << "start[0]: " << _start[0] << std::endl;
        // std::cout << "stop[0]: " << _stop[0] << std::endl;
        // std::cout << "skip[0]: " << _skip[0] << std::endl;

        // std::cout << "count[1]: " << _count[1] << std::endl;
        // std::cout << "start[1]: " << _start[1] << std::endl;
        // std::cout << "stop[1]: " << _stop[1] << std::endl;
        // std::cout << "skip[1]: " << _skip[1] << std::endl;

        return ndarray<rank - 1>(_count, _start, _stop, _skip, data);
    }

    size_t size() const
    {
        auto s = shape();
        return std::accumulate(s.begin(), s.end(), 1, std::multiplies<>());
    }

    std::array<size_t, rank> shape() const
    {
        std::array<size_t, rank> s;

        for (int n = 0; n < rank; ++n)
        {
            s[n] = (stop[n] - start[n]) / skip[n];
        }
        return s;
    }

    std::array<size_t, rank> strides() const
    {
        std::array<size_t, rank> s;
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

    std::shared_ptr<std::vector<double>> data;
    std::array<size_t, rank> count;
    std::array<size_t, rank> start;
    std::array<size_t, rank> stop;
    std::array<size_t, rank> skip;
};
