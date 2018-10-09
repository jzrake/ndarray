#include <memory>
#include <vector>
#include <array>




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
            start[dim] = 0;
            stop[dim] = dim_sizes[dim];
            skip[dim] = 1;
        }
        data = std::make_shared<std::vector<double>>(total_size);
    }

    ndarray(
        std::shared_ptr<std::vector<double>> data,
        std::array<size_t, rank> start,
        std::array<size_t, rank> stop,
        std::array<size_t, rank> skip)
    : data(data)
    , start(start)
    , stop(stop)
    , skip(skip) {}

    ndarray<rank - 1> operator[](int index)
    {
        std::array<size_t, rank - 1> _start;
        std::array<size_t, rank - 1> _stop;
        std::array<size_t, rank - 1> _skip;

        for (int n = 0; n < rank - 1; ++n)
        {
            _start[n] = start[n + 1];
            _stop[n] = stop[n + 1];
            _skip[n] = skip[n + 1];
        }
        return ndarray<rank - 1>(data, _start, _stop, _skip);
    }

    size_t size() const
    {
        return data->size();
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

private:
    size_t offset(std::array<size_t, rank> index) const
    {
        size_t m = 0;
        size_t s = 1;

        for (int n = rank - 1; n >= 0; --n)
        {
            m += s * index[n];
            s *= (stop[n] - start[n]) / skip[n];
        }
        return m;
    }
    std::shared_ptr<std::vector<double>> data;
    std::array<size_t, rank> start;
    std::array<size_t, rank> stop;
    std::array<size_t, rank> skip;
};
