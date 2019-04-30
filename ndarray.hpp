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


    // buffer view support structs
    //=========================================================================
    template<std::size_t Rank, typename ValueType, typename DerivedType> class short_sequence_t;
    template<std::size_t Rank> class shape_t;
    template<std::size_t Rank> class index_t;
    template<std::size_t Rank> class jumps_t;
    template<std::size_t Rank> class access_pattern_t;
    template<typename ValueType> class buffer_t;
    template<std::size_t Rank, typename Provider> class array_t;
    template<std::size_t Rank> class identity_provider_t;


    // algorithm support structs
    //=========================================================================
    template<typename ValueType> class range_container_t;
    template<typename TupleType, typename ValueType> class zipped_container_t;


    // factory functions
    //=========================================================================
    template<typename... Args> auto make_shape(Args... args);
    template<typename... Args> auto make_index(Args... args);
    template<typename... Args> auto make_jumps(Args... args);
    template<std::size_t Rank, typename Arg> auto make_uniform_shape(Arg arg);
    template<std::size_t Rank, typename Arg> auto make_uniform_index(Arg arg);
    template<std::size_t Rank, typename Arg> auto make_uniform_jumps(Arg arg);
    template<std::size_t Rank> auto make_access_pattern(shape_t<Rank> shape);
    template<typename... Args> auto make_access_pattern(Args... args);
    template<typename... Args> auto make_identity_provider(Args... args);
    template<typename Provider> auto make_array(Provider&&);
    template<typename Provider, typename Accessor> auto make_array(Provider&&, Accessor&&);


    // std::algorithm wrappers for ranges
    //=========================================================================
    template<typename Range, typename Seed, typename Function> auto accumulate(Range&& rng, Seed&& seed, Function&& fn);
    template<typename Range, typename Predicate> auto all_of(Range&& rng, Predicate&& pred);
    template<typename Range, typename Predicate> auto any_of(Range&& rng, Predicate&& pred);
    template<typename Range> auto distance(Range&& rng);
    template<typename Range> auto enumerate(Range&& rng);
    template<typename ValueType> auto range(ValueType count);
    template<typename... ContainerTypes> auto zip(ContainerTypes&&... containers);
}




//=============================================================================
template<typename TupleType, typename ValueType>
class nd::zipped_container_t
{
public:
    using value_type = ValueType;

    //=========================================================================
    template<typename IteratorTuple>
    struct iterator
    {
        using iterator_category = std::input_iterator_tag;
        using value_type = ValueType;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        iterator& operator++()
        {
            iterators = transform_tuple([] (auto x) { return ++x; }, iterators);
            return *this;
        }

        bool operator!=(const iterator& other) const
        {
            return iterators != other.iterators;
        }

        auto operator*() const
        {
            return transform_tuple([] (const auto& x) { return std::ref(*x); }, iterators);
        }

        IteratorTuple iterators;
    };

    //=========================================================================
    zipped_container_t(TupleType&& containers) : containers(std::move(containers))
    {
    }

    auto begin() const
    {
        auto res = transform_tuple([] (const auto& x) { return std::begin(x); }, containers);
        return iterator<decltype(res)>{res};
    }

    auto end() const
    {
        auto res = transform_tuple([] (const auto& x) { return std::end(x); }, containers);
        return iterator<decltype(res)>{res};
    }

private:
    //=========================================================================
    template<typename Function, typename Tuple, std::size_t... Is>
    static auto transform_tuple_impl(Function&& fn, const Tuple& t, std::index_sequence<Is...>)
    {
        return std::make_tuple(fn(std::get<Is>(t))...);
    }

    template<typename Function, typename Tuple>
    static auto transform_tuple(Function&& fn, const Tuple& t)
    {
        return transform_tuple_impl(fn, t, std::make_index_sequence<std::tuple_size<Tuple>::value>());
    }

    TupleType containers;
};




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

private:
    //=========================================================================
    ValueType start = 0;
    ValueType final = 0;
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
    using TupleType = std::tuple<ContainerTypes...>;
    using ValueType = std::tuple<typename std::remove_reference_t<ContainerTypes>::value_type...>;
    return nd::zipped_container_t<TupleType, ValueType>(std::forward_as_tuple(containers...));
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

    std::size_t size() const { return accumulate(*this, 1, std::multiplies<>()); }
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

    bool contains(const index_t<Rank>& index) const
    {
        return all_of(zip(index, *this), [] (const auto& t) { return std::get<0>(t) < std::get<1>(t); });
    }

    template<typename... Args>
    bool contains(Args... args) const
    {
        return contains(make_index(args...));
    }
};




//=============================================================================
template<std::size_t Rank>
class nd::index_t : public nd::short_sequence_t<Rank, std::size_t, index_t<Rank>>
{
public:
    using short_sequence_t<Rank, std::size_t, index_t<Rank>>::short_sequence_t;
};




//=============================================================================
template<std::size_t Rank>
class nd::jumps_t : public nd::short_sequence_t<Rank, long, jumps_t<Rank>>
{
public:
    using short_sequence_t<Rank, long, jumps_t<Rank>>::short_sequence_t;
};




//=============================================================================
template<std::size_t Rank>
class nd::access_pattern_t
{
public:

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
        return accumulate(shape(), 1, std::multiplies<int>());
    }

    bool empty() const
    {
        return any_of(shape(), [] (auto s) { return s == 0; });
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

    bool contains(const index_t<Rank>& index) const
    {
        return shape().contains(index);
    }

    template<typename... Args>
    bool contains(Args... args) const
    {
        return contains(make_index(args...));
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

    using value_type = typename Provider::value_type;
    using Accessor = access_pattern_t<Rank>;

    array_t(Provider provider, Accessor accessor) : provider(provider), accessor(accessor) {}

    auto shape() const { return provider.shape(); }
    std::size_t size() const { return accumulate(shape(), 1, std::multiplies<>()); }

    template<typename... Args>
    value_type operator()(Args... args) const
    {
        return provider(accessor.map_index(make_index(args...)));
    }

private:
    Provider provider;
    Accessor accessor;
};




//=========================================================================
template<std::size_t Rank>
class nd::identity_provider_t
{
public:

    using value_type = index_t<Rank>;
    static constexpr std::size_t rank = Rank;

    //=========================================================================
    identity_provider_t(nd::shape_t<Rank> the_shape) : the_shape(the_shape) {}
    auto operator()(const nd::index_t<Rank>& i) const { return i; }
    auto shape() const { return the_shape; }

private:
    //=========================================================================
    nd::shape_t<Rank> the_shape;
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

template<size_t Rank, typename Arg>
auto nd::make_uniform_shape(Arg arg)
{
    return shape_t<Rank>::uniform(arg);
}

template<size_t Rank, typename Arg>
auto nd::make_uniform_index(Arg arg)
{
    return index_t<Rank>::uniform(arg);
}

template<size_t Rank, typename Arg>
auto nd::make_uniform_jumps(Arg arg)
{
    return jumps_t<Rank>::uniform(arg);
}

template<typename... Args> auto nd::make_access_pattern(Args... args)
{
    return access_pattern_t<sizeof...(Args)>().with_final(args...);
}

template<std::size_t Rank>
auto nd::make_access_pattern(shape_t<Rank> shape)
{
    return access_pattern_t<Rank>().with_final(index_t<Rank>::from_range(shape));
}

template<typename... Args> auto nd::make_identity_provider(Args... args)
{
    return identity_provider_t<sizeof...(Args)>(make_shape(args...));
}

template<typename Provider, typename Accessor>
auto nd::make_array(Provider&& provider, Accessor&& accessor)
{
    return array_t<Provider::rank, Provider>(provider, accessor);
}

template<typename Provider>
auto nd::make_array(Provider&& provider)
{
    return array_t<Provider::rank, Provider>(provider, make_access_pattern(provider.shape()));
}




//=============================================================================
template<typename ValueType>
class nd::buffer_t
{
public:

    using value_type = ValueType;

    //=========================================================================
    buffer_t() {}

    ~buffer_t() { delete [] memory; }

    buffer_t(const buffer_t& other)
    {
        count = other.count;
        memory = new ValueType[count];

        for (int n = 0; n < count; ++n)
        {
            memory[n] = other.memory[n];
        }
    }

    buffer_t(buffer_t&& other)
    {
        delete [] memory;
        memory = other.memory;
        count = other.count;

        other.memory = nullptr;
        other.count = 0;
    }

    explicit buffer_t(std::size_t count, const ValueType& value = ValueType())
    : count(count)
    , memory(new ValueType[count])
    {
        for (std::size_t n = 0; n < count; ++n)
        {
            memory[n] = value;
        }
    }

    template<class InputIt>
    buffer_t(InputIt first, InputIt last)
    : count(std::distance(first, last))
    , memory(new ValueType[count])
    {
        for (std::size_t n = 0; n < count; ++n)
        {
            memory[n] = *first++;
        }
    }

    buffer_t& operator=(const buffer_t& other)
    {
        delete [] memory;
        count = other.count;
        memory = new ValueType[count];

        for (int n = 0; n < count; ++n)
        {
            memory[n] = other.memory[n];
        }
        return *this;
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
