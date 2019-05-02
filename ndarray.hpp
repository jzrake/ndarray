#pragma once
#include <algorithm>         // std::all_of
#include <functional>        // std::ref
#include <initializer_list>  // std::initializer_list
#include <iterator>          // std::distance
#include <numeric>           // std::accumulate
#include <utility>           // std::index_sequence
#include <string>            // std::to_string




/**
 * An N-dimensional array is an access pattern (sequence of index<N>), and a
 * provider (shape<N>, index<N> => value_type).
 *
 * The access pattern is based on a start, final, jumps scheme. It has begin/end
 * methods as well as a random-access method which generates an index into the
 * provider, given a source index. Access patterns can be transformed in
 * different ways - selected, shifted, and reshaped. A provider might be able to
 * be reshaped, but not necessarily, and the overlying array can only be
 * reshaped if its provider can. The provider might also offer transformations
 * on its index space, such as permuting axes (including transpositions).
 */




//=============================================================================
namespace nd
{


    // array support structs
    //=========================================================================
    template<std::size_t Rank, typename ValueType, typename DerivedType> class short_sequence_t;
    template<std::size_t Rank, typename Provider> class array_t;
    template<std::size_t Rank> class shape_t;
    template<std::size_t Rank> class index_t;
    template<std::size_t Rank> class jumps_t;
    template<std::size_t Rank> class memory_strides_t;
    template<std::size_t Rank> class access_pattern_t;
    template<typename ValueType> class buffer_t;


    // array and access pattern factory functions
    //=========================================================================
    template<typename... Args> auto make_shape(Args... args);
    template<typename... Args> auto make_index(Args... args);
    template<typename... Args> auto make_jumps(Args... args);
    template<std::size_t Rank, typename Arg> auto make_uniform_shape(Arg arg);
    template<std::size_t Rank, typename Arg> auto make_uniform_index(Arg arg);
    template<std::size_t Rank, typename Arg> auto make_uniform_jumps(Arg arg);
    template<std::size_t Rank> auto make_strides_row_major(shape_t<Rank> shape);
    template<std::size_t Rank> auto make_access_pattern(shape_t<Rank> shape);
    template<typename... Args> auto make_access_pattern(Args... args);


    // provider types
    //=========================================================================
    template<std::size_t Rank> class index_provider_t;
    template<std::size_t Rank, typename ValueType> class uniform_provider_t;
    template<std::size_t Rank, typename ValueType> class shared_provider_t;
    template<std::size_t Rank, typename ValueType> class unique_provider_t;
    template<std::size_t Rank, typename ValueType, typename ArrayTuple> class zipped_provider_t;
    template<typename ArrayType, typename Function> class transform_provider_t;
    template<typename ArrayT, typename ArrayF, typename Predicate> class switch_provider_t;
    template<typename ArrayToPatch, typename ReplacementArray> class replace_provider_t;
    template<typename ArrayType> class select_provider_t;
    // template<std::size_t ReductionInRank> class slice_provider_t;


    // provider factory functions
    //=========================================================================
    template<typename ValueType, std::size_t Rank> auto make_shared_provider(shape_t<Rank> shape);
    template<typename ValueType, typename... Args> auto make_shared_provider(Args... args);
    template<typename ValueType, std::size_t Rank> auto make_unique_provider(shape_t<Rank> shape);
    template<typename ValueType, typename... Args> auto make_unique_provider(Args... args);
    template<typename Provider> auto evaluate_as_shared(Provider&&);
    template<typename Provider> auto evaluate_as_unique(Provider&&);
    template<std::size_t Rank> auto make_index_provider(shape_t<Rank> shape);
    template<typename... Args> auto make_index_provider(Args... args);
    template<typename ValueType, std::size_t Rank> auto make_uniform_provider(ValueType value, shape_t<Rank> shape);
    template<typename ValueType, typename... Args> auto make_uniform_provider(ValueType value, Args... args);
    template<typename... ArrayTypes> auto make_zipped_provider(ArrayTypes&&... arrays);
    template<typename ArrayType, typename Function> auto make_transform_provider(ArrayType&&, Function&&);

    template<typename ArrayT, typename ArrayF, typename Predicate>
    auto make_switch_provider(ArrayT&&, ArrayF&&, Predicate&&);

    template<std::size_t Rank, typename ArrayToPatch, typename ReplacementArray>
    auto make_replace_provider(access_pattern_t<Rank>, ArrayToPatch&&, ReplacementArray&&);

    template<typename ArrayType, std::size_t Rank>
    auto make_select_provider(ArrayType&& provider, access_pattern_t<Rank> accessor);


    // array factory functions
    //=========================================================================
    template<typename Provider> auto make_array(Provider&&);
    template<typename ValueType, std::size_t Rank> auto shared_array(shape_t<Rank> shape);
    template<typename ValueType, typename... Args> auto shared_array(Args... args);
    template<typename ValueType, std::size_t Rank> auto unique_array(shape_t<Rank> shape);
    template<typename ValueType, typename... Args> auto unique_array(Args... args);
    template<std::size_t Rank> auto index_array(shape_t<Rank> shape);
    template<typename... Args> auto index_array(Args... args);
    template<typename... ArrayTypes> auto zip_arrays(ArrayTypes&&... arrays);


    // array operator support structs
    //=========================================================================
    template<std::size_t Rank> class op_reshape_t;
    template<typename ArrayType> class op_replace_t;
    template<typename Function> class op_transform_t;
    template<std::size_t RanksOperatedOn> class op_select_t;
    // template<std::size_t ReductionInRank> class op_slice_t;


    // array operator factory functions
    //=========================================================================
    auto shared();
    auto unique();
    template<std::size_t Rank> auto reshape(shape_t<Rank> shape);
    template<typename... Args> auto reshape(Args... args);
    template<std::size_t Rank, typename ArrayType> auto replace(access_pattern_t<Rank>, ArrayType&&);
    template<std::size_t Rank> auto select(access_pattern_t<Rank>);
    template<typename Function> auto transform(Function&& function);


    // algorithm support structs
    //=========================================================================
    template<typename ValueType> class range_container_t;
    template<typename ValueType, typename ContainerTuple> class zipped_container_t;
    template<typename ContainerType, typename Function> class transformed_container_t;


    // std::algorithm wrappers for ranges
    //=========================================================================
    template<typename Range, typename Seed, typename Function> auto accumulate(Range&& rng, Seed&& seed, Function&& fn);
    template<typename Range, typename Predicate> auto all_of(Range&& rng, Predicate&& pred);
    template<typename Range, typename Predicate> auto any_of(Range&& rng, Predicate&& pred);
    template<typename Range> auto distance(Range&& rng);
    template<typename Range> auto enumerate(Range&& rng);
    template<typename ValueType> auto range(ValueType count);
    template<typename... ContainerTypes> auto zip(ContainerTypes&&... containers);


    // helper functions
    //=========================================================================
    namespace detail
    {
        template<typename Function, typename Tuple, std::size_t... Is>
        auto transform_tuple_impl(Function&& fn, const Tuple& t, std::index_sequence<Is...>);

        template<typename Function, typename Tuple>
        auto transform_tuple(Function&& fn, const Tuple& t);

        template<typename ResultSequence, typename SourceSequence, typename IndexContainer>
        auto remove_elements(const SourceSequence& source, IndexContainer indexes);

        template<typename ResultSequence, typename SourceSequence, typename IndexContainer, typename Sequence>
        auto insert_elements(const SourceSequence& source, IndexContainer indexes, Sequence values);
    }
}




//=============================================================================
template<typename ValueType>
class nd::range_container_t
{
public:

    using value_type = ValueType;

    //=========================================================================
    struct iterator
    {
        using iterator_category = std::input_iterator_tag;
        using value_type = ValueType;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        iterator& operator++() { ++current; return *this; }
        bool operator==(const iterator& other) const { return current == other.current; }
        bool operator!=(const iterator& other) const { return current != other.current; }
        const ValueType& operator*() const { return current; }
        ValueType current = 0;
        ValueType start = 0;
        ValueType final = 0;
    };

    //=========================================================================
    range_container_t(ValueType start, ValueType final) : start(start), final(final) {}
    iterator begin() const { return { 0, start, final }; }
    iterator end() const { return { final, start, final }; }

    template<typename Function>
    auto operator|(Function&& fn) const
    {
        return transformed_container_t<range_container_t, Function>(*this, fn);
    }

private:
    //=========================================================================
    ValueType start = 0;
    ValueType final = 0;
};




//=============================================================================
template<typename ValueType, typename ContainerTuple>
class nd::zipped_container_t
{
public:

    using value_type = ValueType;

    //=========================================================================
    template<typename IteratorTuple>
    struct iterator
    {
        using iterator_category = std::input_iterator_tag;
        using value_type = value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        iterator& operator++()
        {
            iterators = detail::transform_tuple([] (auto x) { return ++x; }, iterators);
            return *this;
        }
        bool operator==(const iterator& other) const { return iterators == other.iterators; }
        bool operator!=(const iterator& other) const { return iterators != other.iterators; }
        auto operator*() const { return detail::transform_tuple([] (const auto& x) { return std::ref(*x); }, iterators); }

        IteratorTuple iterators;
    };

    //=========================================================================
    zipped_container_t(ContainerTuple&& containers) : containers(containers) {}

    auto begin() const
    {
        auto res = detail::transform_tuple([] (const auto& x) { return std::begin(x); }, containers);
         return iterator<decltype(res)>{res};
    }

    auto end() const
    {
        auto res = detail::transform_tuple([] (const auto& x) { return std::end(x); }, containers);
        return iterator<decltype(res)>{res};
    }

    template<typename Function>
    auto operator|(Function&& fn) const
    {
        return transformed_container_t<zipped_container_t, Function>(*this, fn);
    }

private:
    //=========================================================================
    ContainerTuple containers;
};




//=============================================================================
template<typename ContainerType, typename Function>
class nd::transformed_container_t
{
public:
    using value_type = std::invoke_result_t<Function, typename ContainerType::value_type>;

    //=========================================================================
    template<typename IteratorType>
    struct iterator
    {
        using iterator_category = std::input_iterator_tag;
        using value_type = value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        iterator& operator++() { ++current; return *this; }
        bool operator==(const iterator& other) const { return current == other.current; }
        bool operator!=(const iterator& other) const { return current != other.current; }
        auto operator*() const { return function(*current); }

        IteratorType current;
        const Function& function;
    };

    //=========================================================================
    transformed_container_t(const ContainerType& container, const Function& function)
    : container(container)
    , function(function) {}

    auto begin() const { return iterator<decltype(container.begin())> {container.begin(), function}; }
    auto end() const { return iterator<decltype(container.end())> {container.end(), function}; }

private:
    //=========================================================================
    const ContainerType& container;
    const Function& function;
};




//=============================================================================
template<typename Range, typename Seed, typename Function>
auto nd::accumulate(Range&& rng, Seed&& seed, Function&& fn)
{
    return std::accumulate(rng.begin(), rng.end(), std::forward<Seed>(seed), std::forward<Function>(fn));
}

template<typename Range, typename Predicate>
auto nd::all_of(Range&& rng, Predicate&& pred)
{
    return std::all_of(rng.begin(), rng.end(), pred);
}

template<typename Range, typename Predicate>
auto nd::any_of(Range&& rng, Predicate&& pred)
{
    return std::any_of(rng.begin(), rng.end(), pred);
}

template<typename Range>
auto nd::distance(Range&& rng)
{
    return std::distance(rng.begin(), rng.end());
}

template<typename Range>
auto nd::enumerate(Range&& rng)
{
    return zip(range(distance(std::forward<Range>(rng))), std::forward<Range>(rng));
}

template<typename ValueType>
auto nd::range(ValueType count)
{
    return nd::range_container_t<ValueType>(0, count);
}

template<typename... ContainerTypes>
auto nd::zip(ContainerTypes&&... containers)
{
    using ValueType = std::tuple<typename std::remove_reference_t<ContainerTypes>::value_type...>;
    using ContainerTuple = std::tuple<ContainerTypes...>;
    return nd::zipped_container_t<ValueType, ContainerTuple>(std::forward_as_tuple(containers...));
}




//=============================================================================
template<std::size_t Rank, typename ValueType, typename DerivedType>
class nd::short_sequence_t
{
public:

    using value_type = ValueType;

    //=========================================================================
    static DerivedType uniform(ValueType arg)
    {
        DerivedType result;

        for (auto n : range(Rank))
        {
            result.memory[n] = arg;
        }
        return result;
    }

    template<typename Range>
    static DerivedType from_range(Range&& rng)
    {
        if (distance(rng) != Rank)
        {
            throw std::logic_error("sequence constructed from range of wrong size");
        }
        DerivedType result;

        for (const auto& [n, a] : enumerate(rng))
        {
            result.memory[n] = a;
        }
        return result;
    }

    short_sequence_t()
    {
        for (auto n : range(Rank))
        {
            memory[n] = ValueType();
        }
    }

    short_sequence_t(std::initializer_list<ValueType> args)
    {
        for (const auto& [n, a] : enumerate(args))
        {
            memory[n] = a;
        }
    }

    bool operator==(const DerivedType& other) const
    {
        return all_of(zip(*this, other), [] (const auto& t) { return std::get<0>(t) == std::get<1>(t); });
    }

    bool operator!=(const DerivedType& other) const
    {
        return any_of(zip(*this, other), [] (const auto& t) { return std::get<0>(t) != std::get<1>(t); });
    }

    constexpr std::size_t size() const { return Rank; }
    const ValueType* data() const { return memory; }
    const ValueType* begin() const { return memory; }
    const ValueType* end() const { return memory + Rank; }
    const ValueType& operator[](std::size_t n) const { return memory[n]; }
    ValueType* data() { return memory; }
    ValueType* begin() { return memory; }
    ValueType* end() { return memory + Rank; }
    ValueType& operator[](std::size_t n) { return memory[n]; }

private:
    //=========================================================================
    ValueType memory[Rank];
};




//=============================================================================
template<std::size_t Rank>
class nd::shape_t : public nd::short_sequence_t<Rank, std::size_t, shape_t<Rank>>
{
public:
    using short_sequence_t<Rank, std::size_t, shape_t<Rank>>::short_sequence_t;

    std::size_t volume() const { return accumulate(*this, 1, std::multiplies<>()); }

    bool contains(const index_t<Rank>& index) const
    {
        return all_of(zip(index, *this), [] (const auto& t) { return std::get<0>(t) < std::get<1>(t); });
    }

    template<typename... Args>
    bool contains(Args... args) const
    {
        return contains(make_index(args...));
    }

    template<typename IndexContainer>
    auto remove_elements(IndexContainer indexes) const
    {
        return detail::remove_elements<shape_t<Rank - indexes.size()>>(*this, indexes);
    }

    template<typename IndexContainer, typename Sequence>
    auto insert_elements(IndexContainer indexes, Sequence values) const
    {
        return detail::insert_elements<shape_t<Rank + indexes.size()>>(*this, indexes, values);
    }

    index_t<Rank> last_index() const
    {
        auto result = index_t<Rank>();

        for (std::size_t n = 0; n < Rank; ++n)
        {
            result[n] = this->operator[](n);
        }
        return result;
    }
};




//=============================================================================
template<std::size_t Rank>
class nd::index_t : public nd::short_sequence_t<Rank, std::size_t, index_t<Rank>>
{
public:
    using short_sequence_t<Rank, std::size_t, index_t<Rank>>::short_sequence_t;

    template<typename IndexContainer>
    auto remove_elements(IndexContainer indexes) const
    {
        return detail::remove_elements<index_t<Rank - indexes.size()>>(*this, indexes);
    }

    template<typename IndexContainer, typename Sequence>
    auto insert_elements(IndexContainer indexes, Sequence values) const
    {
        return detail::insert_elements<index_t<Rank + indexes.size()>>(*this, indexes, values);
    }

    bool operator<(const index_t<Rank>& other) const
    {
        return all_of(zip(*this, other), [] (const auto& t) { return std::get<0>(t) < std::get<1>(t); });
    }
    bool operator>(const index_t<Rank>& other) const
    {
        return all_of(zip(*this, other), [] (const auto& t) { return std::get<0>(t) > std::get<1>(t); });
    }
    bool operator<=(const index_t<Rank>& other) const
    {
        return all_of(zip(*this, other), [] (const auto& t) { return std::get<0>(t) <= std::get<1>(t); });
    }
    bool operator>=(const index_t<Rank>& other) const
    {
        return all_of(zip(*this, other), [] (const auto& t) { return std::get<0>(t) >= std::get<1>(t); });
    }
};




//=============================================================================
template<std::size_t Rank>
class nd::jumps_t : public nd::short_sequence_t<Rank, long, jumps_t<Rank>>
{
public:
    using short_sequence_t<Rank, long, jumps_t<Rank>>::short_sequence_t;

    template<typename IndexContainer>
    auto remove_elements(IndexContainer indexes) const
    {
        return detail::remove_elements<jumps_t<Rank - indexes.size()>>(*this, indexes);
    }

    template<typename IndexContainer, typename Sequence>
    auto insert_elements(IndexContainer indexes, Sequence values) const
    {
        return detail::insert_elements<jumps_t<Rank + indexes.size()>>(*this, indexes, values);
    }
};




//=============================================================================
template<std::size_t Rank>
class nd::memory_strides_t : public nd::short_sequence_t<Rank, std::size_t, memory_strides_t<Rank>>
{
public:
    using short_sequence_t<Rank, std::size_t, memory_strides_t<Rank>>::short_sequence_t;

    std::size_t compute_offset(const index_t<Rank>& index) const
    {
        auto mul_tuple = [] (auto t) { return std::get<0>(t) * std::get<1>(t); };
        return accumulate(zip(index, *this) | mul_tuple, 0, std::plus<>());
    }

    template<typename... Args>
    std::size_t compute_offset(Args... args) const
    {
        return compute_offset(make_index(args...));
    }
};




//=============================================================================
template<std::size_t Rank>
class nd::access_pattern_t
{
public:

    using value_type = index_t<Rank>;
    static constexpr std::size_t rank = Rank;

    //=========================================================================
    struct iterator
    {
        using iterator_category = std::input_iterator_tag;
        using value_type = index_t<Rank>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        iterator& operator++() { accessor.advance(current); return *this; }
        bool operator==(const iterator& other) const { return current == other.current; }
        bool operator!=(const iterator& other) const { return current != other.current; }
        const index_t<Rank>& operator*() const { return current; }

        access_pattern_t accessor;
        index_t<Rank> current;
    };

    //=========================================================================
    template<typename... Args> access_pattern_t with_start(Args... args) const { return { make_index(args...), final, jumps }; }
    template<typename... Args> access_pattern_t with_final(Args... args) const { return { start, make_index(args...), jumps }; }
    template<typename... Args> access_pattern_t with_jumps(Args... args) const { return { start, final, make_jumps(args...) }; }

    access_pattern_t with_start(index_t<Rank> arg) const { return { arg, final, jumps }; }
    access_pattern_t with_final(index_t<Rank> arg) const { return { start, arg, jumps }; }
    access_pattern_t with_jumps(jumps_t<Rank> arg) const { return { start, final, arg }; }

    std::size_t size() const
    {
        return shape().volume();
    }

    auto shape() const
    {
        auto s = shape_t<Rank>();

        for (std::size_t n = 0; n < Rank; ++n)
        {
            s[n] = final[n] / jumps[n] - start[n] / jumps[n];
        }
        return s;
    }

    bool empty() const
    {
        return any_of(shape(), [] (auto s) { return s == 0; });
    }

    bool contiguous() const
    {
        return
        start == make_uniform_index<Rank>(0) &&
        jumps == make_uniform_jumps<Rank>(1);
    }

    bool operator==(const access_pattern_t& other) const
    {
        return
        start == other.start &&
        final == other.final &&
        jumps == other.jumps;
    }

    bool operator!=(const access_pattern_t& other) const
    {
        return
        start != other.start ||
        final != other.final ||
        jumps != other.jumps;
    }

    bool advance(index_t<Rank>& index) const
    {
        int n = Rank - 1;

        index[n] += jumps[n];

        while (index[n] >= final[n])
        {
            if (n == 0)
            {
                index = final;
                return false;
            }
            index[n] = start[n];

            --n;

            index[n] += jumps[n];
        }
        return true;
    }

    index_t<Rank> map_index(const index_t<Rank>& index) const
    {
        index_t<Rank> result;

        for (std::size_t n = 0; n < Rank; ++n)
        {
            result[n] = start[n] + jumps[n] * index[n];
        }
        return result;
    }

    index_t<Rank> inverse_map_index(const index_t<Rank>& mapped_index) const
    {
        index_t<Rank> result;

        for (std::size_t n = 0; n < Rank; ++n)
        {
            result[n] = (mapped_index[n] - start[n]) / jumps[n];
        }
        return result;
    }

    /**
     * Return true if this is a valid mapped-from index.
     */
    bool contains(const index_t<Rank>& index) const
    {
        return shape().contains(index);
    }

    template<typename... Args>
    bool contains(Args... args) const
    {
        return contains(make_index(args...));
    }

    /**
     * Return true if an iteration over this accessor would generate the given
     * index, that is, if it the index included in the set of mapped-to indexes.
     */
    bool generates(const index_t<Rank>& mapped_index) const
    {
        for (std::size_t n = 0; n < Rank; ++n)
        {
            if ((mapped_index[n] <  start[n]) ||
                (mapped_index[n] >= final[n]) ||
                (mapped_index[n] -  start[n]) % jumps[n] != 0)
            {
                return false;
            }
        }
        return true;
    }

    template<typename... Args>
    bool generates(Args... args) const
    {
        return generates(make_index(args...));
    }

    /**
     * Return false if this access pattern would generate any indexes not
     * contained in the given shape.
     */
    bool within(const shape_t<Rank>& parent_shape) const
    {
        auto zero = make_uniform_index<Rank>(0);
        auto t1 = map_index(zero);
        auto t2 = map_index(shape().last_index());

        return (t1 >= zero && t1 <= parent_shape.last_index() &&
                t2 >= zero && t2 <= parent_shape.last_index());
    }

    iterator begin() const { return { *this, start }; }
    iterator end() const { return { *this, final }; }

    //=========================================================================
    index_t<Rank> start = make_uniform_index<Rank>(0);
    index_t<Rank> final = make_uniform_index<Rank>(0);
    jumps_t<Rank> jumps = make_uniform_jumps<Rank>(1);
};




//=============================================================================
template<std::size_t Rank>
class nd::op_reshape_t
{
public:
    op_reshape_t(shape_t<Rank> new_shape) : new_shape(new_shape) {}

    template<typename Array>
    auto operator()(Array&& array) const
    {
        const auto& provider = array.get_provider();

        if (new_shape.volume() != provider.size())
        {
            throw std::logic_error("cannot reshape array to a different size");
        }
        return make_array(provider.reshape(new_shape));
    }

private:
    //=========================================================================
    shape_t<Rank> new_shape;
};




//=============================================================================
template<typename ArrayType>
class nd::op_replace_t
{
public:

    static constexpr std::size_t rank = std::remove_reference_t<ArrayType>::rank;

    //=========================================================================
    op_replace_t(access_pattern_t<rank> region_to_replace, ArrayType&& replacement_array)
    : region_to_replace(region_to_replace)
    , replacement_array(replacement_array)
    {
        if (region_to_replace.shape() != replacement_array.shape())
        {
            throw std::logic_error("region to replace has a different shape than the replacement array");
        }
    }

    template<typename ArrayToPatch>
    auto operator()(ArrayToPatch&& array_to_patch) const
    {
        return make_array(make_replace_provider(region_to_replace, array_to_patch, replacement_array));
    }

private:
    //=========================================================================
    access_pattern_t<rank> region_to_replace;
    ArrayType replacement_array;
};




//=============================================================================
template<typename Function>
class nd::op_transform_t
{
public:

    //=========================================================================
    op_transform_t(Function&& function) : function(function) {}

    template<typename ArgumentArray>
    auto operator()(ArgumentArray&& argument_array) const
    {
        return make_array(make_transform_provider(argument_array, function));
    }

private:
    //=========================================================================
    Function function;
};




//=============================================================================
template<std::size_t Rank>
class nd::op_select_t
{
public:

    //=========================================================================
    op_select_t(access_pattern_t<Rank> accessor) : accessor(accessor) {}

    template<typename ArgumentArray>
    auto operator()(ArgumentArray&& array) const
    {
        return make_array(make_select_provider(array, accessor));
    }

private:
    //=========================================================================
    access_pattern_t<Rank> accessor;
};




//=============================================================================
template<std::size_t Rank, typename Provider>
class nd::array_t
{
public:

    using provider_type = Provider;
    using value_type = typename Provider::value_type;
    static constexpr std::size_t rank = Rank;

    //=========================================================================
    array_t(Provider&& provider) : provider(std::move(provider)) {}

    // indexing functions
    //=========================================================================
    template<typename... Args> decltype(auto) operator()(Args... args) const { return provider(make_index(args...)); }
    template<typename... Args> decltype(auto) operator()(Args... args)       { return provider(make_index(args...)); }
    decltype(auto) operator()(const index_t<Rank>& index) const { return provider(index); }
    decltype(auto) operator()(const index_t<Rank>& index)       { return provider(index); }

    // query functions and operator support
    //=========================================================================
    auto shape() const { return provider.shape(); }
    auto size() const { return provider.size(); }
    const Provider& get_provider() const { return provider; }
    auto get_accessor() { return make_access_pattern(provider.shape()); }
    template<typename Function> auto operator|(Function&& fn) const & { return fn(*this); }
    template<typename Function> auto operator|(Function&& fn)      && { return fn(std::move(*this)); }

    // methods converting this to a memory-backed array
    //=========================================================================
    auto unique() const { return make_array(evaluate_as_unique(provider)); }
    auto shared() const { return make_array(evaluate_as_shared(provider)); }

private:
    //=========================================================================
    Provider provider;
};




//=============================================================================
template<std::size_t Rank>
class nd::index_provider_t
{
public:

    using value_type = index_t<Rank>;
    static constexpr std::size_t rank = Rank;

    //=========================================================================
    index_provider_t(shape_t<Rank> the_shape) : the_shape(the_shape) {}
    auto operator()(const index_t<Rank>& index) const
    {
        if (! the_shape.contains(index))
        {
            throw std::out_of_range("index out-of-range on index_provider");
        }
        return index;
    }
    auto shape() const { return the_shape; }
    auto size() { return the_shape.volume(); }

private:
    //=========================================================================
    shape_t<Rank> the_shape;
};




//=============================================================================
template<std::size_t Rank, typename ValueType>
class nd::uniform_provider_t
{
public:

    using value_type = ValueType;
    static constexpr std::size_t rank = Rank;

    //=========================================================================
    uniform_provider_t(shape_t<Rank> the_shape, ValueType the_value) : the_shape(the_shape), the_value(the_value) {}
    const ValueType& operator()(const index_t<Rank>&) const { return the_value; }
    auto shape() const { return the_shape; }
    auto size() { return the_shape.volume(); }
    template<std::size_t NewRank> auto reshape(shape_t<NewRank> new_shape) const { return uniform_provider_t<NewRank, ValueType>(new_shape, the_value); }

private:
    //=========================================================================
    shape_t<Rank> the_shape;
    ValueType the_value;
};




//=============================================================================
template<std::size_t Rank, typename ValueType>
class nd::shared_provider_t
{
public:

    using value_type = ValueType;
    static constexpr std::size_t rank = Rank;

    //=========================================================================
    shared_provider_t(nd::shape_t<Rank> the_shape, std::shared_ptr<nd::buffer_t<ValueType>> buffer)
    : the_shape(the_shape)
    , strides(make_strides_row_major(the_shape))
    , buffer(buffer)
    {
        if (the_shape.volume() != buffer->size())
        {
            throw std::logic_error("shape and buffer sizes do not match");
        }
    }

    const ValueType& operator()(const index_t<Rank>& index) const
    {
        return buffer->operator[](strides.compute_offset(index));
    }

    auto shape() const { return the_shape; }
    auto size() const { return the_shape.volume(); }
    const ValueType* data() const { return buffer->data(); }

    template<std::size_t NewRank> auto reshape(shape_t<NewRank> new_shape) const { return shared_provider_t<NewRank, ValueType>(new_shape, buffer); }

private:
    //=========================================================================
    shape_t<Rank> the_shape;
    memory_strides_t<Rank> strides;
    std::shared_ptr<buffer_t<ValueType>> buffer;
};




//=============================================================================
template<std::size_t Rank, typename ValueType>
class nd::unique_provider_t
{
public:

    using value_type = ValueType;
    static constexpr std::size_t rank = Rank;

    //=========================================================================
    unique_provider_t(nd::shape_t<Rank> the_shape, nd::buffer_t<ValueType>&& buffer)
    : the_shape(the_shape)
    , strides(make_strides_row_major(the_shape))
    , buffer(std::move(buffer))
    {
        if (the_shape.volume() != unique_provider_t::buffer.size())
        {
            throw std::logic_error("shape and buffer sizes do not match");
        }
    }

    const ValueType& operator()(const index_t<Rank>& index) const { return buffer.operator[](strides.compute_offset(index)); }
    /* */ ValueType& operator()(const index_t<Rank>& index)       { return buffer.operator[](strides.compute_offset(index)); }
    template<typename... Args> const ValueType& operator()(Args... args) const { return operator()(make_index(args...)); }
    template<typename... Args> /* */ ValueType& operator()(Args... args)       { return operator()(make_index(args...)); }

    auto shape() const { return the_shape; }
    auto size() const { return the_shape.volume(); }
    const ValueType* data() const { return buffer.data(); }

    auto shared() const & { return shared_provider_t(the_shape, std::make_shared<buffer_t<ValueType>>(buffer.begin(), buffer.end())); }
    auto shared()      && { return shared_provider_t(the_shape, std::make_shared<buffer_t<ValueType>>(std::move(buffer))); }

    template<std::size_t NewRank> auto reshape(shape_t<NewRank> new_shape) const & { return unique_provider_t<NewRank, ValueType>(new_shape, buffer_t<ValueType>(buffer.begin(), buffer.end())); }
    template<std::size_t NewRank> auto reshape(shape_t<NewRank> new_shape)      && { return unique_provider_t<NewRank, ValueType>(new_shape, std::move(buffer)); }

private:
    //=========================================================================
    shape_t<Rank> the_shape;
    memory_strides_t<Rank> strides;
    buffer_t<ValueType> buffer;
};




//=============================================================================
template<std::size_t Rank, typename ValueType, typename ArrayTuple>
class nd::zipped_provider_t
{
public:

    using value_type = ValueType;
    static constexpr std::size_t rank = Rank;

    //=========================================================================
    zipped_provider_t(shape_t<Rank> the_shape, ArrayTuple&& arrays)
    : the_shape(the_shape)
    , arrays(arrays) {}

    auto operator()(const index_t<Rank>& index) const { return detail::transform_tuple([index] (auto&& A) { return A(index); }, arrays); }
    auto shape() const { return the_shape; }
    auto size() const { return the_shape.volume(); }

private:
    shape_t<Rank> the_shape;
    ArrayTuple arrays;
};




//=============================================================================
template<typename ArrayType, typename Function>
class nd::transform_provider_t
{
public:

    using argument_value_type = typename std::remove_reference_t<ArrayType>::value_type;
    using value_type = std::invoke_result_t<Function, argument_value_type>;
    static constexpr std::size_t rank = std::remove_reference_t<ArrayType>::rank;

    //=========================================================================
    transform_provider_t(ArrayType&& argument_array, Function&& function)
    : argument_array(argument_array)
    , function(function) {}

    auto operator()(const index_t<rank>& index) const { return function(argument_array(index)); }
    auto shape() const { return argument_array.shape(); }
    auto size() const { return argument_array.size(); }

private:
    ArrayType argument_array;
    Function function;
};




//=============================================================================
template<typename ArrayT, typename ArrayF, typename Predicate>
class nd::switch_provider_t
{
public:

    using value_type = typename std::remove_reference_t<ArrayT>::value_type;
    static constexpr std::size_t rank = std::remove_reference_t<ArrayT>::rank;

    //=========================================================================
    switch_provider_t(shape_t<rank> the_shape, ArrayT&& array1, ArrayF&& array2, Predicate&& predicate)
    : the_shape(the_shape)
    , array1(array1)
    , array2(array2)
    , predicate(predicate) {}

    auto operator()(const index_t<rank>& index) const { return predicate(index) ? array1(index) : array2(index); }
    auto shape() const { return the_shape; }
    auto size() const { return the_shape.volume(); }

private:
    shape_t<rank> the_shape;
    ArrayT array1;
    ArrayF array2;
    Predicate predicate;
};




//=============================================================================
template<typename ArrayToPatch, typename ReplacementArray>
class nd::replace_provider_t
{
public:
    using value_type = typename std::remove_reference_t<ArrayToPatch>::value_type;
    static constexpr std::size_t rank = std::remove_reference_t<ArrayToPatch>::rank;


    //=========================================================================
    replace_provider_t(
        access_pattern_t<rank> patched_region,
        ArrayToPatch&& array_to_patch,
        ReplacementArray&& replacement_array)
    : patched_region(patched_region)
    , array_to_patch(array_to_patch)
    , replacement_array(replacement_array) {}

    auto operator()(const index_t<rank>& index) const
    {
        if (patched_region.generates(index))
        {
            return replacement_array(patched_region.inverse_map_index(index));
        }
        return array_to_patch(index);
    }

    auto shape() const { return array_to_patch.shape(); }
    auto size() const { return array_to_patch.size(); }

private:
    //=========================================================================
    access_pattern_t<rank> patched_region;
    ArrayToPatch array_to_patch;
    ReplacementArray replacement_array;
};




//=============================================================================
template<typename ArrayType>
class nd::select_provider_t
{
public:

    using value_type = typename std::remove_reference_t<ArrayType>::value_type;
    static constexpr std::size_t rank = std::remove_reference_t<ArrayType>::rank;

    //=========================================================================
    select_provider_t(ArrayType&& provider, access_pattern_t<rank> accessor)
    : provider(provider)
    , accessor(accessor)
    {
        if (! accessor.within(provider.shape()))
        {
            throw std::logic_error("out-of-bounds selection");
        }
    }

    decltype(auto) operator()(const index_t<rank>& index) const
    {
        return provider(accessor.map_index(index));
    }

    auto shape() const { return accessor.shape(); }
    auto size() const { return accessor.size(); }

private:
    //=========================================================================
    ArrayType provider;
    access_pattern_t<rank> accessor;
};




//=============================================================================
template<typename ValueType>
class nd::buffer_t
{
public:

    using value_type = ValueType;

    //=========================================================================
    ~buffer_t() { delete [] memory; }
    buffer_t() {}
    buffer_t(const buffer_t& other) = delete;
    buffer_t& operator=(const buffer_t& other) = delete;

    // buffer_t(const buffer_t& other)
    // {
    //     memory = new ValueType[other.count];
    //     count = other.count;

    //     for (std::size_t n = 0; n < count; ++n)
    //     {
    //         memory[n] = other.memory[n];
    //     }
    // }

    // buffer_t& operator=(const buffer_t& other)
    // {
    //     delete [] memory;
    //     count = other.count;
    //     memory = new ValueType[count];

    //     for (std::size_t n = 0; n < count; ++n)
    //     {
    //         memory[n] = other.memory[n];
    //     }
    //     return *this;
    // }

    buffer_t(buffer_t&& other)
    {
        memory = other.memory;
        count = other.count;
        other.memory = nullptr;
        other.count = 0;
    }

    buffer_t(std::size_t count, ValueType value=ValueType())
    : count(count)
    , memory(new ValueType[count])
    {
        for (std::size_t n = 0; n < count; ++n)
        {
            memory[n] = value;
        }
    }

    template<class IteratorType>
    buffer_t(IteratorType first, IteratorType last)
    : count(std::distance(first, last)), memory(new ValueType[count])
    {
        for (std::size_t n = 0; n < count; ++n)
        {
            memory[n] = *first++;
        }
    }

    buffer_t& operator=(buffer_t&& other)
    {
        delete [] memory;
        memory = other.memory;
        count = other.count;

        other.memory = nullptr;
        other.count = 0;
        return *this;
    }

    bool operator==(const buffer_t& other) const
    {
        return count == other.count
        && all_of(zip(*this, other), [] (const auto& t) { return std::get<0>(t) == std::get<1>(t); });
    }

    bool operator!=(const buffer_t& other) const
    {
        return count != other.count
        || any_of(zip(*this, other), [] (const auto& t) { return std::get<0>(t) != std::get<1>(t); });
    }

    bool empty() const { return count == 0; }
    std::size_t size() const { return count; }

    const ValueType* data() const { return memory; }
    const ValueType* begin() const { return memory; }
    const ValueType* end() const { return memory + count; }
    const ValueType& operator[](std::size_t offset) const { return memory[offset]; }
    const ValueType& at(std::size_t offset) const
    {
        if (offset >= count)
        {
            throw std::out_of_range("buffer_t index out of range on index "
                + std::to_string(offset) + " / "
                + std::to_string(count));
        }
        return memory[offset];
    }

    ValueType* data() { return memory; }
    ValueType* begin() { return memory; }
    ValueType* end() { return memory + count; }
    ValueType& operator[](std::size_t offset) { return memory[offset]; }
    ValueType& at(std::size_t offset)
    {
        if (offset >= count)
        {
            throw std::out_of_range("buffer_t index out of range on index "
                + std::to_string(offset) + " / "
                + std::to_string(count));
        }
        return memory[offset];
    }

private:
    //=========================================================================
    std::size_t count = 0;
    ValueType* memory = nullptr;
};




//=============================================================================
template<typename... Args>
auto nd::make_shape(Args... args)
{
    return shape_t<sizeof...(Args)>({std::size_t(args)...});
}

template<typename... Args>
auto nd::make_index(Args... args)
{
    return index_t<sizeof...(Args)>({std::size_t(args)...});
}

template<typename... Args>
auto nd::make_jumps(Args... args)
{
    return jumps_t<sizeof...(Args)>({long(args)...});
}

template<std::size_t Rank, typename Arg>
auto nd::make_uniform_shape(Arg arg)
{
    return shape_t<Rank>::uniform(arg);
}

template<std::size_t Rank, typename Arg>
auto nd::make_uniform_index(Arg arg)
{
    return index_t<Rank>::uniform(arg);
}

template<std::size_t Rank, typename Arg>
auto nd::make_uniform_jumps(Arg arg)
{
    return jumps_t<Rank>::uniform(arg);
}

template<std::size_t Rank>
auto nd::make_strides_row_major(shape_t<Rank> shape)
{
    auto result = memory_strides_t<Rank>();

    result[Rank - 1] = 1;

    if constexpr (Rank > 1)
    {
        for (int n = Rank - 2; n >= 0; --n)
        {
            result[n] = result[n + 1] * shape[n + 1];
        }
    }
    return result;
}

template<std::size_t Rank>
auto nd::make_access_pattern(shape_t<Rank> shape)
{
    return access_pattern_t<Rank>().with_final(index_t<Rank>::from_range(shape));
}

template<typename... Args>
auto nd::make_access_pattern(Args... args)
{
    return access_pattern_t<sizeof...(Args)>().with_final(args...);
}




//=============================================================================
template<std::size_t Rank>
auto nd::make_index_provider(shape_t<Rank> shape)
{
    return index_provider_t<Rank>(shape);
}

template<typename... Args>
auto nd::make_index_provider(Args... args)
{
    return make_index_provider(make_shape(args...));
}

template<typename ValueType, std::size_t Rank>
auto nd::make_uniform_provider(ValueType value, shape_t<Rank> shape)
{
    return uniform_provider_t<Rank, ValueType>(shape, value);
}

template<typename ValueType, typename... Args>
auto nd::make_uniform_provider(ValueType value, Args... args)
{
    return make_uniform_provider(value, make_shape(args...));
}

template<typename ValueType, std::size_t Rank>
auto nd::make_shared_provider(shape_t<Rank> shape)
{
    auto buffer = std::make_shared<buffer_t<ValueType>>(shape.volume());
    return shared_provider_t<Rank, ValueType>(shape, buffer);
}

template<typename ValueType, typename... Args>
auto nd::make_shared_provider(Args... args)
{
    return make_shared_provider<ValueType>(make_shape(args...));
}

template<typename ValueType, std::size_t Rank>
auto nd::make_unique_provider(shape_t<Rank> shape)
{
    auto buffer = buffer_t<ValueType>(shape.volume());
    return unique_provider_t<Rank, ValueType>(shape, std::move(buffer));
}

template<typename ValueType, typename... Args>
auto nd::make_unique_provider(Args... args)
{
    return make_unique_provider<ValueType>(make_shape(args...));
}

template<typename... ArrayTypes>
auto nd::make_zipped_provider(ArrayTypes&&... arrays)
{
    using ValueType = std::tuple<typename ArrayTypes::value_type...>;
    using ArrayTuple = std::tuple<ArrayTypes...>;
    constexpr std::size_t Ranks[] = {ArrayTypes::rank...};
    shape_t<Ranks[0]> shapes[] = {arrays.shape()...};

    if (std::adjacent_find(std::begin(shapes), std::end(shapes), std::not_equal_to<>()) != std::end(shapes))
    {
        throw std::logic_error("cannot zip arrays with different shapes");
    }
    return zipped_provider_t<Ranks[0], ValueType, ArrayTuple>(shapes[0], std::forward_as_tuple(arrays...));
}

template<typename ArrayType, typename Function>
auto nd::make_transform_provider(ArrayType&& argument_array, Function&& function)
{
    return transform_provider_t<ArrayType, Function>(argument_array, function);
}

template<typename ArrayT, typename ArrayF, typename Predicate>
auto nd::make_switch_provider(ArrayT&& array1, ArrayF&& array2, Predicate&& predicate)
{
    if (array1.shape() != array2.shape())
    {
        throw std::logic_error("ambiguous shape for switch provider");
    }
    return switch_provider_t<ArrayT, ArrayF, Predicate>(
        array1.shape(),
        std::forward<ArrayT>(array1),
        std::forward<ArrayF>(array2),
        std::forward<Predicate>(predicate));
}

template<std::size_t Rank, typename ArrayToPatch, typename ReplacementArray>
auto nd::make_replace_provider(
    access_pattern_t<Rank> patched_region,
    ArrayToPatch&& array_to_patch,
    ReplacementArray&& replacement_array)
{
    return replace_provider_t<ArrayToPatch, ReplacementArray>(
        patched_region,
        std::forward<ArrayToPatch>(array_to_patch),
        std::forward<ReplacementArray>(replacement_array));
}

template<typename ArrayType, std::size_t Rank>
auto nd::make_select_provider(ArrayType&& provider, access_pattern_t<Rank> accessor)
{
    return select_provider_t<ArrayType>(provider, accessor);
}

template<typename Provider>
auto nd::evaluate_as_unique(Provider&& source_provider)
{
    using value_type = typename std::remove_reference_t<Provider>::value_type;
    auto target_shape = source_provider.shape();
    auto target_accessor = make_access_pattern(target_shape);
    auto target_provider = make_unique_provider<value_type>(target_shape);

    for (const auto& index : target_accessor)
    {
        target_provider(index) = source_provider(index);
    }
    return target_provider;
}

template<typename Provider>
auto nd::evaluate_as_shared(Provider&& provider)
{
    return evaluate_as_unique(std::forward<Provider>(provider)).shared();
}




//=============================================================================
template<typename Provider>
auto nd::make_array(Provider&& provider)
{
    return array_t<Provider::rank, Provider>(std::forward<Provider>(provider));
}

template<typename ValueType, std::size_t Rank>
auto nd::shared_array(shape_t<Rank> shape)
{
    return make_array(make_shared_provider<ValueType>(shape));
}

template<typename ValueType, typename... Args>
auto nd::shared_array(Args... args)
{
    return make_array(make_shared_provider<ValueType>(args...));
}

template<typename ValueType, std::size_t Rank>
auto nd::unique_array(shape_t<Rank> shape)
{
    return make_array(make_unique_provider<ValueType>(shape));
}

template<typename ValueType, typename... Args>
auto nd::unique_array(Args... args)
{
    return make_array(make_unique_provider<ValueType>(args...));
}

template<std::size_t Rank>
auto nd::index_array(shape_t<Rank> shape)
{
    return make_array(make_index_provider(shape));
}

template<typename... Args>
auto nd::index_array(Args... args)
{
    return make_array(make_index_provider(args...));
}

template<typename... ArrayTypes>
auto nd::zip_arrays(ArrayTypes&&... arrays)
{
    auto provider = make_zipped_provider(std::forward<ArrayTypes>(arrays)...);
    auto shape = provider.shape();
    return make_array(std::move(provider), shape);
}




//=============================================================================
auto nd::shared()
{
    return [] (auto&& array)
    {
        return make_array(evaluate_as_unique(array.get_provider()));
    };
}

auto nd::unique()
{
    return [] (auto&& array)
    {
        return make_array(evaluate_as_shared(array.get_provider()));
    };
}

template<std::size_t Rank>
auto nd::reshape(shape_t<Rank> shape)
{
    return op_reshape_t<Rank>(shape);
}

template<typename... Args>
auto nd::reshape(Args... args)
{
    return reshape(make_shape(args...));
}

template<std::size_t Rank, typename ArrayType>
auto nd::replace(access_pattern_t<Rank> region_to_replace, ArrayType&& replacement_array)
{
    return op_replace_t<ArrayType>(region_to_replace, std::forward<ArrayType>(replacement_array));
}

template<std::size_t Rank>
auto nd::select(access_pattern_t<Rank> accessor)
{
    return op_select_t<Rank>(accessor);
}

template<typename Function>
auto nd::transform(Function&& function)
{
    return op_transform_t<Function>(std::forward<Function>(function));
}




//=============================================================================
template<typename Function, typename Tuple, std::size_t... Is>
auto nd::detail::transform_tuple_impl(Function&& fn, const Tuple& t, std::index_sequence<Is...>)
{
    return std::make_tuple(fn(std::get<Is>(t))...);
}

template<typename Function, typename Tuple>
auto nd::detail::transform_tuple(Function&& fn, const Tuple& t)
{
    return transform_tuple_impl(fn, t, std::make_index_sequence<std::tuple_size<Tuple>::value>());
}

template<typename ResultSequence, typename SourceSequence, typename IndexContainer>
auto nd::detail::remove_elements(const SourceSequence& source, IndexContainer indexes)
{
    auto target_n = std::size_t(0);
    auto result = ResultSequence();

    for (std::size_t n = 0; n < source.size(); ++n)
    {
        if (std::find(std::begin(indexes), std::end(indexes), n) == std::end(indexes))
        {
            result[target_n++] = source[n];
        }
    }
    return result;
}

template<typename ResultSequence, typename SourceSequence, typename IndexContainer, typename Sequence>
auto nd::detail::insert_elements(const SourceSequence& source, IndexContainer indexes, Sequence values)
{
    static_assert(indexes.size() == values.size());

    auto source1_n = std::size_t(0);
    auto source2_n = std::size_t(0);
    auto result = ResultSequence();

    for (std::size_t n = 0; n < result.size(); ++n)
    {
        if (std::find(std::begin(indexes), std::end(indexes), n) == std::end(indexes))
        {
            result[n] = source[source1_n++];
        }
        else
        {
            result[n] = values[source2_n++];
        }
    }
    return result;
}
