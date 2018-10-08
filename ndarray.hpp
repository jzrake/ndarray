#include <memory>
#include <vector>
#include <array>




// ============================================================================
template <int rank>
class ndarray
{
public:

  template <typename... Dims>
  ndarray(Dims... dims)
  : ndarray(std::array<int, rank>({dims...}))
  {
    static_assert(sizeof...(dims) == rank,
      "Number of arguments to ndarray constructor must match rank");
  }

  ndarray(std::array<int, rank> dim_sizes)
  {
    size_t total_size = 1;

    for (auto dim_size : dim_sizes)
    {
      total_size *= dim_size;
    }
    data = std::make_shared<std::vector<double>>(total_size);

    for (auto dim = 0; dim < rank; ++dim)
    {
      start[dim] = 0;
      stop[dim] = dim_sizes[dim];
      skip[dim] = 1;
    }
  }

  ndarray<rank-1> operator[](int index)
  {
    return ndarray<rank-1>(1, 2);
  }

private:
  std::array<size_t, rank> start;
  std::array<size_t, rank> stop;
  std::array<size_t, rank> skip;
  std::shared_ptr<std::vector<double>> data;
};




int main()
{
  ndarray<3> A(1, 2, 3);
}
