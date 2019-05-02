#pragma once
#include <algorithm>         // std::all_of
#include <functional>        // std::ref
#include <initializer_list>  // std::initializer_list
#include <iterator>          // std::distance
#include <numeric>           // std::accumulate
#include <utility>           // std::index_sequence
#include <string>            // std::to_string




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
    template<typename Function, std::size_t Rank> class basic_provider_t;
    template<std::size_t Rank, typename ValueType> class uniform_provider_t;
    template<std::size_t Rank, typename ValueType> class shared_provider_t;
    template<std::size_t Rank, typename ValueType> class unique_provider_t;


    // provider factory functions
    //=========================================================================
    template<typename ValueType, std::size_t Rank> auto make_shared_provider(shape_t<Rank> shape);
    template<typename ValueType, typename... Args> auto make_shared_provider(Args... args);
    template<typename ValueType, std::size_t Rank> auto make_unique_provider(shape_t<Rank> shape);
    template<typename ValueType, typename... Args> auto make_unique_provider(Args... args);
    template<typename ValueType, std::size_t Rank> auto make_uniform_provider(ValueType value, shape_t<Rank> shape);
    template<typename ValueType, typename... Args> auto make_uniform_provider(ValueType value, Args... args);
    template<typename Provider> auto evaluate_as_shared(Provider&&);
    template<typename Provider> auto evaluate_as_unique(Provider&&);


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
    constexpr std::size_t get_rank() { return Rank; }
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
template<typename Function, std::size_t Rank>
class nd::basic_provider_t
{
public:

    using value_type = std::invoke_result_t<Function, index_t<Rank>>;
    static constexpr std::size_t rank = Rank;

    //=========================================================================
    basic_provider_t(Function mapping, shape_t<Rank> the_shape) : mapping(mapping), the_shape(the_shape) {}
    decltype(auto) operator()(const index_t<Rank>& index) const { return mapping(index); }
    auto shape() const { return the_shape; }
    auto size() const { return the_shape.volume(); }

private:
    //=========================================================================
    Function mapping;
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
// Shape, index, and access pattern factories
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
// Provider factories
//=============================================================================




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
// Array factories
//=============================================================================




/**
 * @brief      Makes an array from the given provider.
 *
 * @param      provider  The provider
 *
 * @tparam     Provider  The type of the provider
 *
 * @return     The array
 */
template<typename Provider>
auto nd::make_array(Provider&& provider)
{
    return array_t<Provider::rank, Provider>(std::forward<Provider>(provider));
}





/**
 * @brief      Makes a shared (immutable, copyable, memory-backed) array with
 *             the given shape, initialized to the default-constructed
 *             ValueType.
 *
 * @param[in]  shape      The shape
 *
 * @tparam     ValueType  The value type of the array
 * @tparam     Rank       The rank of the array
 *
 * @return     The array
 */
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





/**
 * @brief      Makes a unique (mutable, non-copyable, memory-backed) array with
 *             the given shape.
 *
 * @param[in]  shape      The shape
 *
 * @tparam     ValueType  The value type of the array
 * @tparam     Rank       The rank of the array
 *
 * @return     The array
 */
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





/**
 * @brief      Returns an index-array of the given shape, mapping the index (i,
 *             j, ...) to itself.
 *
 * @param[in]  shape  The shape
 *
 * @tparam     Rank   The rank of the array
 *
 * @return     The array
 */
template<std::size_t Rank>
auto nd::index_array(shape_t<Rank> shape)
{
    auto mapping = [shape] (auto&& index)
    {
        if (! shape.contains(index))
        {
            throw std::out_of_range("out-of-range on index array");
        }
        return index;
    };
    return make_array(basic_provider_t<decltype(mapping), Rank>(mapping, shape));
}

template<typename... Args>
auto nd::index_array(Args... args)
{
    return index_array(make_shape(args...));
}




/**
 * @brief      Zip a sequence identically-shaped arrays together
 *
 * @param      arrays      The arrays
 *
 * @tparam     ArrayTypes  The types of the arrays
 *
 * @return     An array which returns tuples taken from the underlying arrays
 */
template<typename... ArrayTypes>
auto nd::zip_arrays(ArrayTypes&&... arrays)
{
    constexpr std::size_t Ranks[] = {std::remove_reference_t<ArrayTypes>::rank...};
    shape_t<Ranks[0]> shapes[] = {arrays.shape()...};

    if (std::adjacent_find(std::begin(shapes), std::end(shapes), std::not_equal_to<>()) != std::end(shapes))
    {
        throw std::logic_error("cannot zip arrays with different shapes");
    }

    auto mapping = [arrays...] (auto&& index) { return std::make_tuple(arrays(index)...); };

    return make_array(basic_provider_t<decltype(mapping), Ranks[0]>(mapping, shapes[0]));
}




//=============================================================================
// Operator factories
//=============================================================================




/**
 * @brief      Return an operator that attempts to reshape its argument array to
 *             the given shape.
 *
 * @param[in]  new_shape  The new shape
 *
 * @tparam     Rank       The rank of the argument array
 *
 * @return     The operator
 */
template<std::size_t Rank>
auto nd::reshape(shape_t<Rank> new_shape)
{
    return [new_shape] (auto&& array)
    {
        const auto& provider = array.get_provider();

        if (new_shape.volume() != provider.size())
        {
            throw std::logic_error("cannot reshape array to a different size");
        }
        return make_array(provider.reshape(new_shape));
    };
}

template<typename... Args>
auto nd::reshape(Args... args)
{
    return reshape(make_shape(args...));
}




/**
 * @brief      Returns an operator that, applied to any array will yield a
 *             shared, memory-backed version of that array.
 *
 * @return     The operator.
 */
auto nd::shared()
{
    return [] (auto&& array)
    {
        return make_array(evaluate_as_unique(array.get_provider()));
    };
}




/**
 * @brief      Returns an operator that, applied to any array will yield a
 *             unique, memory-backed version of that array.
 *
 * @return     The operator.
 */
auto nd::unique()
{
    return [] (auto&& array)
    {
        return make_array(evaluate_as_shared(array.get_provider()));
    };
}




/**
 * @brief      Replace a subset of an array with the contents of another.
 *
 * @param[in]  region_to_replace  The region to replace
 * @param      replacement_array  The replacement array
 *
 * @tparam     Rank               Rank of both the array to patch and the
 *                                replacement array
 * @tparam     ArrayType          The type of the replacement array
 *
 * @return     A function returning arrays which map their indexes to the
 *             replacement_array, if those indexes are in the region to replace
 */
template<std::size_t Rank, typename ArrayType>
auto nd::replace(access_pattern_t<Rank> region_to_replace, ArrayType&& replacement_array)
{
    if (region_to_replace.shape() != replacement_array.shape())
    {
        throw std::logic_error("region to replace has a different shape than the replacement array");
    }

    return [region_to_replace, replacement_array] (auto&& array_to_patch)
    {
        auto mapping = [region_to_replace, replacement_array, array_to_patch] (auto&& index)
        {
            if (region_to_replace.generates(index))
            {
                return replacement_array(region_to_replace.inverse_map_index(index));
            }
            return array_to_patch(index);
        };
        auto shape = array_to_patch.shape();
        return make_array(basic_provider_t<decltype(mapping), Rank>(mapping, shape));
    };
}




/**
 * @brief      Return an operator that selects a subset of an array.
 *
 * @param[in]  region_to_select  The region to select
 *
 * @tparam     Rank              Rank of the both the source and target arrays
 *
 * @return     The operator.
 */
template<std::size_t Rank>
auto nd::select(nd::access_pattern_t<Rank> region_to_select)
{
    return [region_to_select] (auto&& array)
    {
        if (! region_to_select.within(array.shape()))
        {
            throw std::logic_error("out-of-bounds selection");
        }
        auto mapping = [array, region_to_select] (auto&& index) { return array(region_to_select.map_index(index)); };
        auto shape = region_to_select.shape();
        return make_array(basic_provider_t<decltype(mapping), Rank>(mapping, shape));
    };
}





/**
 * @brief      Return an operator that transforms the values of an array using
 *             the given function object.
 *
 * @param      function  The function
 *
 * @tparam     Function  The type of the function object
 *
 * @return     The operator
 */
template<typename Function>
auto nd::transform(Function&& function)
{
    return [function] (auto&& array)
    {
        constexpr std::size_t Rank = std::remove_reference_t<decltype(array)>::rank;
        auto mapping = [array, function] (auto&& index) { return function(array(index)); };
        return make_array(basic_provider_t<decltype(mapping), Rank>(mapping, array.shape()));
    };
}




//=============================================================================
// Helper functions
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
