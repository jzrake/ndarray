#include <memory>
#include <vector>
#include <array>
#include <numeric>




// ============================================================================
template<int Rank, int Axis=0>
struct selector
{
    enum { rank = Rank, axis = Axis };

    /**
     * Collapse this selector at the given index, creating a selector with
     * rank reduced by 1, and which operates on the same axis.
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

    selector<rank - 1, axis> within(int start_index) const
    {
        return collapse(start_index);
    }

    selector<rank, axis + 1> within(std::tuple<int, int> range) const
    {
        static_assert(axis < rank, "selector: attempting to index on axis >= rank");

        auto _count = count;
        auto _start = start;
        auto _final = final;

        _start[axis] = start[axis] + std::get<0>(range);
        _final[axis] = start[axis] + std::get<1>(range);

        return {_count, _start, _final};
    }

    template<typename First, typename... Rest>
    auto within(First first, Rest... rest) const
    {
        return within(first).within(rest...);
    }

    selector<rank> reset() const
    {
        return selector<rank>{count, start, final};
    }

    std::array<int, rank> shape() const
    {
        std::array<int, rank> s;

        for (int n = 0; n < rank; ++n)
        {
            s[n] = final[n] - start[n];
        }
        return s;
    }

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

    std::array<int, rank> count;
    std::array<int, rank> start;
    std::array<int, rank> final;
};




// ============================================================================
template<int rank>
class ndarray
{
public:




    /**
     * Constructors
     * 
     */
    // ========================================================================
    template<int R = rank, typename = typename std::enable_if<R == 0>::type>
    ndarray(double value=0.0)
    : scalar_offset(0)
    , data(std::make_shared<std::vector<double>>(1, value))
    {
    }

    template<int R = rank, typename = typename std::enable_if<R == 0>::type>
    ndarray(int scalar_offset, std::shared_ptr<std::vector<double>> data)
    : scalar_offset(scalar_offset)
    , data(data)
    {
    }

    template<typename... Dims>
    ndarray(Dims... dims) : ndarray(std::array<int, rank>({dims...}))
    {
        static_assert(sizeof...(dims) == rank,
          "Number of arguments to ndarray constructor must match rank");
    }

    template<typename SelectorType>
    ndarray(SelectorType selector, std::shared_ptr<std::vector<double>> data)
    : count(selector.count)
    , start(selector.start)
    , final(selector.final)
    , skips(constant_array<int, rank>(1))
    , strides(compute_strides(count))
    , data(data)
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

    ndarray(std::array<int, rank> dim_sizes, std::shared_ptr<std::vector<double>> data)
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
        std::shared_ptr<std::vector<double>> data)
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
     * Shape and size query methods
     * 
     */
    // ========================================================================
    int size() const
    {
        return make_selector().size();
    }
    int shape() const
    {
        return make_selector().shape();
    }




    /**
     * Data accessors and selection methods
     * 
     */
    // ========================================================================
    template <int R = rank, typename std::enable_if_t<R == 1>* = nullptr>
    ndarray<rank - 1> operator[](int index)
    {
        return ndarray<0>(offset({index}), data);
    }

    template <int R = rank, typename std::enable_if_t<R != 1>* = nullptr>
    ndarray<rank - 1> operator[](int index)
    {
        return {make_selector().collapse(index), data};
    }

    template<typename... Index>
    double& operator()(Index... index)
    {
        static_assert(sizeof...(index) == rank,
          "Number of arguments to ndarray::operator() must match rank");

        return data->operator[](offset({index...}));
    }

    operator double() const
    {
        static_assert(rank == 0, "can only convert rank-0 array to scalar value");
        return data->operator[](0);
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

    template<int other_rank>
    bool shares(const ndarray<other_rank>& other) const
    {
        return data == other.data;
    }




private:
    /**
     * Pivate utility methods
     * 
     */
    // ========================================================================
    int offset(std::array<int, rank> index) const
    {
        int m = scalar_offset;

        for (int n = 0; n < rank; ++n)
        {
            m += start[n] + skips[n] * index[n] * strides[n];
        }
        return m;
    }

    static std::array<int, rank> compute_strides(std::array<int, rank> count)
    {
        if (rank == 0)
        {
            return std::array<int, rank>();
        }
        std::array<int, rank> s;   
        std::partial_sum(count.rbegin(), count.rend() - 1, s.rbegin() + 1, std::multiplies<>());
        s[rank - 1] = 1;
        return s;
    }

    selector<rank> make_selector() const
    {
        return selector<rank>{count, start, final};
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




    /** Data members
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




    /** Grant friendship to ndarray's of other ranks.
     *
     */
    template<int other_rank>
    friend class ndarray;
};









// // ============================================================================
// template<int rank>
// class ndarray
// {
// public:




//     // ========================================================================
//     template<typename S, int a>
//     struct recurse_selector
//     {
//         static ndarray<rank> select(ndarray<rank> A, S s)
//         {
//             return recurse_selector<S, a - 1>::select(A.select(a, std::get<0>(s[a]), std::get<1>(s[a])), s);
//         }
//     };

//     template<typename S>
//     struct recurse_selector<S, 0>
//     {
//         static ndarray<rank> select(ndarray<rank> A, S s)
//         {
//             return A.select(0, std::get<0>(s[0]), std::get<1>(s[0]));
//         }
//     };




//     // ========================================================================
//     template<typename... Dims>
//     ndarray(Dims... dims) : ndarray(std::array<int, rank>({dims...}))
//     {
//         static_assert(sizeof...(dims) == rank,
//           "Number of arguments to ndarray constructor must match rank");
//     }

//     ndarray(std::array<int, rank> dim_sizes)
//     : count(dim_sizes)
//     , start(constant_int_array<rank>(0))
//     , stop(dim_sizes)
//     , skip(constant_int_array<rank>(1))
//     , data(std::make_shared<std::vector<double>>(product(dim_sizes)))
//     {}

//     ndarray(std::array<int, rank> dim_sizes, std::shared_ptr<std::vector<double>> data)
//     : count(dim_sizes)
//     , start(constant_int_array<rank>(0))
//     , stop(dim_sizes)
//     , skip(constant_int_array<rank>(1))
//     , data(data)
//     {
//         assert(data->size() == std::accumulate(count.begin(), count.end(), 1, std::multiplies<>()));
//     }

//     ndarray(
//         std::array<int, rank> count,
//         std::array<int, rank> start,
//         std::array<int, rank> stop,
//         std::array<int, rank> skip,
//         std::shared_ptr<std::vector<double>> data)
//     : count(count)
//     , start(start)
//     , stop(stop)
//     , skip(skip)
//     , data(data)
//     {
//         assert(data->size() == std::accumulate(count.begin(), count.end(), 1, std::multiplies<>()));
//     }




//     // ========================================================================
//     ndarray<rank - 1> operator[](int index)
//     {
//         return collapse(0, index);
//     }

//     ndarray<rank - 1> collapse(int axis, int start_index) const
//     {
//         auto s = strides();

//         std::array<int, rank - 1> _count;
//         std::array<int, rank - 1> _start;
//         std::array<int, rank - 1> _stop;
//         std::array<int, rank - 1> _skip;

//         for (int n = 0; n < axis; ++n)
//         {
//             _count[n] = count[n];
//             _start[n] = start[n];
//             _stop[n] = stop[n];
//             _skip[n] = skip[n];
//         }

//         for (int n = axis + 1; n < rank - 1; ++n)
//         {
//             _count[n] = count[n + 1];
//             _start[n] = start[n + 1];
//             _stop[n] = stop[n + 1];
//             _skip[n] = skip[n + 1];
//         }

//         _count[axis] = count[axis] * count[axis + 1];
//         _start[axis] = start[axis] * s[axis] + skip[axis] * start_index * s[axis];
//         _stop[axis] = _start[axis] + count[axis + 1];
//         _skip[axis] = skip[axis + 1];

//         return ndarray<rank - 1>(_count, _start, _stop, _skip, data);
//     }

//     ndarray<rank> select(int axis, int start_index, int stop_index) const
//     {
//         auto _count = count;
//         auto _start = start;
//         auto _stop = stop;
//         auto _skip = skip;

//         _start[axis] = start[axis] + start_index;
//         _stop[axis] = start[axis] + stop_index;

//         return ndarray<rank>(_count, _start, _stop, _skip, data);
//     }

//     template<typename... Slices>
//     ndarray<rank> within(Slices... slices) const
//     {
//         static_assert(sizeof...(slices) == rank,
//           "Number of arguments to ndarray::within must match rank");

//         return recurse_selector<
//         std::array<std::tuple<int, int>, rank>,
//         rank - 1>::select(*this, {slices...});
//     }

//     size_t size() const
//     {
//         auto s = shape();
//         return std::accumulate(s.begin(), s.end(), 1, std::multiplies<>());
//     }

//     std::array<int, rank> shape() const
//     {
//         std::array<int, rank> s;

//         for (int n = 0; n < rank; ++n)
//         {
//             s[n] = (stop[n] - start[n]) / skip[n];
//         }
//         return s;
//     }

//     std::array<int, rank> strides() const
//     {
//         std::array<int, rank> s = {1};
//         std::partial_sum(count.rbegin(), count.rend() - 1, s.rbegin() + 1, std::multiplies<int>());
//         s[rank - 1] = 1;
//         return s;
//     }

//     template<typename... Index>
//     size_t offset(Index... index) const
//     {
//         static_assert(sizeof...(index) == rank,
//           "Number of arguments to ndarray::offset must match rank");

//         return offset({index...});
//     }

//     template<typename... Index>
//     double& operator()(Index... index)
//     {
//         static_assert(sizeof...(index) == rank,
//           "Number of arguments to ndarray::operator() must match rank");

//         return data->operator[](offset({index...}));
//     }

//     template<typename... Index>
//     const double& operator()(Index... index) const
//     {
//         static_assert(sizeof...(index) == rank,
//           "Number of arguments to ndarray::operator() must match rank");

//         return data->operator[](offset({index...}));
//     }

//     template<int new_rank>
//     ndarray<new_rank> reshape(std::array<int, new_rank> dim_sizes) const
//     {
//         assert(contiguous());
//         return ndarray<new_rank>(dim_sizes, data);
//     }

//     template<typename... DimSizes>
//     ndarray<sizeof...(DimSizes)> reshape(DimSizes... dim_sizes) const
//     {
//         assert(contiguous());
//         return ndarray<sizeof...(DimSizes)>({dim_sizes...}, data);
//     }

//     bool contiguous() const
//     {
//         for (int n = 0; n < rank; ++n)
//         {
//             if (start[n] != 0 || stop[n] != count[n] || skip[n] != 1)
//             {
//                 return false;
//             }
//         }
//         return true;
//     }

//     template<int other_rank>
//     bool shares(const ndarray<other_rank>& other) const
//     {
//         return data == other.data;
//     }

//     /**
//      * Private methods
//      * 
//      */
// private:
//     size_t offset(std::array<int, rank> index) const
//     {
//         size_t m = 0;
//         size_t s = 1;

//         for (int n = rank - 1; n >= 0; --n)
//         {
//             m += start[n] * s + skip[n] * index[n] * s;
//             s *= count[n];
//         }
//         return m;
//     }

//     template<int length>
//     static std::array<int, length> constant_int_array(int value)
//     {
//         std::array<int, length> A;

//         for (auto& a : A)
//         {
//             a = value;
//         }
//         return A;
//     }

//     template<typename C>
//     static size_t product(const C& c)
//     {
//         return std::accumulate(c.begin(), c.end(), 1, std::multiplies<>());
//     }

//     std::shared_ptr<std::vector<double>> data;
//     std::array<int, rank> count;
//     std::array<int, rank> start;
//     std::array<int, rank> stop;
//     std::array<int, rank> skip;

//     /** Grants friendship to ndarrays of all other ranks.
//      */
//     template<int other_rank>
//     friend class ndarray;
// };
