#include <iostream>
#include <cassert>
#include "ndarray.hpp"



/**
    Below is some code I got from StackExchange that supposedly allows to filter
    a variadic template parameter pack.

    https://codereview.stackexchange.com/questions/115740/filtering-variadic-template-arguments
*/
// ============================================================================

// This works: but you can't instantiate list
// template <typename...> struct list;

// This also works: you can now instantiate a list
template <typename... I>
using list = std::tuple<I...>;

template <typename... Lists> struct concat;

template <template<typename> class Pred, typename T>
using filter_helper = std::conditional_t<Pred<T>::value, list<T>, list<>>;

template <template <typename> class Pred, typename... Ts> 
using filter = typename concat<filter_helper<Pred, Ts>...>::type;

template <>
struct concat<> { using type = list<>; };

template <typename... Ts>
struct concat<list<Ts...>> { using type = list<Ts...>; };

template <typename... Ts, typename... Us>
struct concat<list<Ts...>, list<Us...>> { using type = list<Ts..., Us...>; };

template<typename... Ts, typename... Us, typename... Rest>
struct concat<list<Ts...>, list<Us...>, Rest...> { using type = typename concat<list<Ts..., Us...>, Rest...>::type; };




template<typename A>
class type_is_int
{
public:
    static constexpr bool value = std::is_same<A, int>::value;
};



void test_filter()
{
    using F = filter<type_is_int, int, int>;
    static_assert(std::is_same<F, list<int, int>>::value, "");


    using G = filter<type_is_int, int, double>;
    static_assert(std::is_same<G, list<int>>::value, "");

    G g;
}




// ============================================================================
int main()
{
    assert(ndarray<3>(2, 3, 4).size() == 24);
    assert(ndarray<3>(2, 3, 4).shape() == (std::array<int, 3>{2, 3, 4}));

    assert(ndarray<2>(3, 3)[0].shape() == (std::array<int, 1>{3}));
    assert(ndarray<2>(3, 3)[1].shape() == (std::array<int, 1>{3}));
    assert(ndarray<2>(3, 3)[1].strides() == (std::array<int, 1>{1}));

    assert(ndarray<3>(3, 4, 5)[0].shape() == (std::array<int, 2>{4, 5}));
    assert(ndarray<3>(3, 4, 5)[1].shape() == (std::array<int, 2>{4, 5}));
    assert(ndarray<3>(3, 4, 5)[0].strides() == (std::array<int, 2>{5, 1}));

    assert(ndarray<3>(2, 3, 4).offset(0, 0, 0) == 0);
    assert(ndarray<3>(2, 3, 4).offset(0, 0, 1) == 1);
    assert(ndarray<3>(2, 3, 4).offset(0, 1, 0) == 4);
    assert(ndarray<3>(2, 3, 4).offset(0, 2, 1) == 9);
    assert(ndarray<3>(2, 3, 4).offset(1, 2, 1) == 21);


	ndarray<3> A(3, 3, 3);
	ndarray<3> B(3, 3, 3);
    assert(A.shares(A) == true);
    assert(A.shares(B) == false);
    assert(A.shares(A.reshape(9, 3)) == true);

    A(0, 0, 0) = 12;
    std::cout << A(0, 0, 0) << std::endl;

    //A.select({0, 1, 10}, {}, 0, {});
}
