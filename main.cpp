// #include "catch.hpp"
// #include "ndarray.hpp"




// // ============================================================================
// TEST_CASE("selector<4> passes basic sanity checks", "[selector]")
// {
//     auto S = selector<4>{{4, 3, 2, 3}, {0, 0, 0, 0}, {4, 3, 2, 3}, {1, 1, 1, 1}};

//     REQUIRE(S.axis == 0);
//     REQUIRE(S.count == std::array<int, 4>{4, 3, 2, 3});
//     REQUIRE(S.shape() == S.count);
// }


// TEST_CASE("selector<4> has correct shape and size if non-contiguous", "[selector]")
// {
//     auto S0 = selector<4>{{4, 3, 8, 5}, {0, 0, 0, 0}, {4, 3, 8, 5}, {1, 1, 1, 1}};
//     auto S1 = selector<4>{{4, 3, 8, 5}, {0, 0, 0, 0}, {4, 3, 8, 5}, {4, 1, 2, 1}};
//     auto S2 = selector<4>{{4, 3, 8, 5}, {0, 0, 0, 0}, {4, 3, 8, 5}, {1, 3, 1, 1}};
//     auto S3 = selector<4>{{4, 3, 8, 5}, {0, 0, 0, 0}, {4, 3, 8, 5}, {1, 3, 1, 2}};

//     REQUIRE(S0.size() == 480);
//     REQUIRE(S1.size() == 60);
//     REQUIRE(S2.size() == 160);
//     REQUIRE(S3.size() == 64);
//     REQUIRE(S0.shape() == std::array<int, 4>{4, 3, 8, 5});
//     REQUIRE(S1.shape() == std::array<int, 4>{1, 3, 4, 5});
//     REQUIRE(S2.shape() == std::array<int, 4>{4, 1, 8, 5});
//     REQUIRE(S3.shape() == std::array<int, 4>{4, 1, 8, 2});
// }


// TEST_CASE("selector<1> works when instantiated as a subset", "[selector]")
// {
//     auto S = selector<1>{{10}, {2}, {8}, {1}};

//     REQUIRE(S.axis == 0);
//     REQUIRE(S.count == std::array<int, 1>{10});
//     REQUIRE(S.shape() == std::array<int, 1>{6});
// }


// TEST_CASE("selector<2> works when instantiated as a subset", "[selector]")
// {
//     auto S = selector<2>{{10, 12}, {2, 4}, {8, 6}, {1, 1}};

//     REQUIRE(S.axis == 0);
//     REQUIRE(S.count == std::array<int, 2>{10, 12});
//     REQUIRE(S.shape() == std::array<int, 2>{6, 2});
//     REQUIRE(S.size() == 12);
// }


// TEST_CASE("selector<2> collapses properly to selector<1>", "[selector::collapse]")
// {
//     auto S = selector<2>{{10, 12}, {0, 0}, {10, 12}, {1, 1}};

//     REQUIRE(S.collapse(0).axis == 0);
//     REQUIRE(S.collapse(0).count == std::array<int, 1>{120});
//     REQUIRE(S.collapse(0).shape() == std::array<int, 1>{12});
//     REQUIRE(S.collapse(0).size() == 12);
// }


// TEST_CASE("selector<2> subset collapses properly to selector<1>", "[selector::collapse]")
// {
//     auto S = selector<2>{{10, 12}, {2, 4}, {8, 6}, {1, 1}};

//     REQUIRE(S.collapse(0).axis == 0);
//     REQUIRE(S.collapse(0).count == std::array<int, 1>{120});
//     REQUIRE(S.collapse(0).start == std::array<int, 1>{24});
//     REQUIRE(S.collapse(0).final == std::array<int, 1>{26});
//     REQUIRE(S.collapse(0).size() == 2);

//     REQUIRE(S.collapse(1).axis == 0);
//     REQUIRE(S.collapse(1).count == std::array<int, 1>{120});
//     REQUIRE(S.collapse(1).start == std::array<int, 1>{36});
//     REQUIRE(S.collapse(1).final == std::array<int, 1>{38});
//     REQUIRE(S.collapse(1).size() == 2);

//     REQUIRE(S.collapse(2).axis == 0);
//     REQUIRE(S.collapse(2).count == std::array<int, 1>{120});
//     REQUIRE(S.collapse(2).start == std::array<int, 1>{48});
//     REQUIRE(S.collapse(2).final == std::array<int, 1>{50});
//     REQUIRE(S.collapse(2).size() == 2);
// }


// TEST_CASE("selector<3> collapses properly to selector<2> on axes 1, 2", "[selector::collapse]")
// {
//     auto S = selector<3>{{10, 12, 14}, {0, 0, 0}, {10, 12, 14}, {1, 1, 1}};

//     REQUIRE(S.on<0>().collapse(0).axis == 0);
//     REQUIRE(S.on<0>().collapse(0).count == std::array<int, 2>{10 * 12, 14});
//     REQUIRE(S.on<0>().collapse(0).start == std::array<int, 2>{0, 0});
//     REQUIRE(S.on<0>().collapse(0).final == std::array<int, 2>{12, 14});
//     //REQUIRE(S.on<0>().collapse(0).size() == 12 * 14);
// }


// TEST_CASE("selector<3> combines properly to selector<2> on axes 1, 2", "[selector::combine]")
// {
//     auto S = selector<3>{{10, 12, 14}, {0, 0, 0}, {10, 12, 14}, {1, 1, 1}};

//     REQUIRE(S.on<0>().combine().rank == 2);
//     REQUIRE(S.on<0>().combine().axis == 0);
//     REQUIRE(S.on<0>().combine().count == std::array<int, 2>{10 * 12, 14});
//     REQUIRE(S.on<0>().combine().skips == std::array<int, 2>{1, 1});
//     REQUIRE(S.on<0>().combine().shape() == std::array<int, 2>{10 * 12, 14});

//     REQUIRE(S.on<1>().combine().rank == 2);
//     REQUIRE(S.on<1>().combine().axis == 1);
//     REQUIRE(S.on<1>().combine().count == std::array<int, 2>{10, 12 * 14});
//     REQUIRE(S.on<1>().combine().skips == std::array<int, 2>{1, 1});
//     REQUIRE(S.on<1>().combine().shape() == std::array<int, 2>{10, 12 * 14});
// }


// TEST_CASE("selector<2> subset is created properly from a selector<2>", "[selector::select]")
// {
//     auto S = selector<2>{{10, 12}, {0, 0}, {10, 12}, {1, 1}};
//     REQUIRE(S.select(std::make_tuple(0, 10)).reset() == S);
//     REQUIRE(S.select(std::make_tuple(2, 4)).reset() == selector<2>{{10, 12}, {2, 0}, {4, 12}, {1, 1}});
//     REQUIRE(S.select(std::make_tuple(2, 8)).reset().select(std::make_tuple(2, 4)).reset() == selector<2>{{10, 12}, {4, 0}, {6, 12}, {1, 1}});
// }


// TEST_CASE("selector<1> next advances properly", "[selector::next]")
// {
//     auto S = selector<1>{{10}, {0}, {10}, {1}};
//     auto I = std::array<int, 1>{0};
//     auto i = 0;

//     do {
//         REQUIRE(i == I[0]);
//         ++i;
//     } while (S.next(I));
// }


// TEST_CASE("selector<2> next advances properly", "[selector::next]")
// {
//     auto S = selector<2>{{10, 10}, {0, 0}, {10, 10}, {1, 1}};
//     auto I = std::array<int, 2>{0, 0};
//     auto i = 0;
//     auto j = 0;

//     do {
//         REQUIRE(i == I[0]);
//         REQUIRE(j == I[1]);

//         if (++j == 10)
//         {
//             j = 0;
//             ++i;
//         }
//     } while (S.next(I));
// }


// TEST_CASE("selector<2> subset next advances properly", "[selector::next]")
// {
//     auto S = selector<2>{{10, 10}, {2, 4}, {8, 6}, {1, 1}};
//     auto I = std::array<int, 2>{2, 4};
//     auto i = 2;
//     auto j = 4;

//     do {
//         REQUIRE(i == I[0]);
//         REQUIRE(j == I[1]);

//         if (++j == 6)
//         {
//             j = 4;
//             ++i;
//         }
//     } while (S.next(I));
// }


// TEST_CASE("selector<2> subset iterator passes sanity checks", "[selector::iterator]")
// {
//     auto S = selector<2>{{10, 10}, {2, 4}, {8, 6}, {1, 1}};
//     auto I = S.start;

//     for (auto index : S)
//     {
//         REQUIRE(index == I);
//         S.next(I);
//     }
// }


// TEST_CASE("scalar ndarray (ndarray<0>) passes sanity checks", "[ndarray]")
// {
//     auto A = ndarray<0>(3.14);
//     REQUIRE(A.rank == 0);
//     REQUIRE(A() == 3.14);
//     REQUIRE(A == 3.14);

//     A() = 2.0;

//     REQUIRE(A() == 2.0);
//     REQUIRE(A == 2.0);
// }


// TEST_CASE("ndarray<1> passes sanity checks", "[ndarray]")
// {
//     auto A = ndarray<1>{0, 1, 2, 3, 4};
//     REQUIRE(A.rank == 1);
//     REQUIRE(A.size() == 5);
//     REQUIRE(A.shape() == std::array<int, 1>{5});
//     REQUIRE(A(0) == 0);
//     REQUIRE(A(4) == 4);
//     REQUIRE(A[0] == 0);
//     REQUIRE(A[4] == 4);
//     REQUIRE(A.is(A));
//     REQUIRE(! A.copy().is(A));
// }


// TEST_CASE("ndarray<1> iterator passes sanity checks", "[ndarray::iterator]")
// {
//     auto A = ndarray<1>{0, 1, 2, 3, 4};
//     auto x = 0.0;

//     REQUIRE(A.begin() == A.begin());
//     REQUIRE(A.begin() != A.end());

//     SECTION("conventional iterator works")
//     {
//         for (auto it = A.begin(); it != A.end(); ++it)
//         {
//             REQUIRE(*it == x++);
//         }
//     }
//     SECTION("range-based for loop works")
//     {
//         for (auto y : A)
//         {
//             REQUIRE(y == x++);
//         }
//     }
// }


// TEST_CASE("ndarray<3> can be sliced, iterated on, and loaded into std::vector", "[ndarray::iterator]")
// {
//     auto A = ndarray<3>(10, 30, 2);

//     for (auto& a : A[5])
//     {
//         a = 5.0;
//     }

//     auto vector_data = std::vector<double>(A[5].begin(), A[5].end());

//     REQUIRE(vector_data.size() == A[5].size());

//     for (auto d : vector_data)
//     {
//         REQUIRE(d == 5.0);
//     }
// }


// TEST_CASE("ndarray<3> can be sliced on all dimensions (except last)", "[ndarray::select]")
// {
//     auto I = [] (auto i, auto j) { return std::make_tuple(i, j); };
//     auto A = ndarray<3>(10, 30, 2);

//     for (int i = 0; i < A.shape()[0]; ++i)
//     {
//         REQUIRE(A[i].shape() == std::array<int, 2>{30, 2});
//         REQUIRE(A.select(0, I(0, 30), I(0, 2)).shape() == std::array<int, 2>{30, 2});
//     }

//     for (int j = 0; j < A.shape()[1]; ++j)
//     {
//         REQUIRE(A.select(I(0, 10),  0, I(0, 2)).shape() == std::array<int, 2>{10, 2});
//         REQUIRE(A.select(I(0, 10),  1, I(0, 2)).shape() == std::array<int, 2>{10, 2});
//         REQUIRE(A.select(I(0, 10), 29, I(0, 2)).shape() == std::array<int, 2>{10, 2});
//     }

//     for (int k = 0; k < A.shape()[2]; ++k)
//     {
//         // REQUIRE(A.select(I(0, 10), I(0, 30), 0).shape() == std::array<int, 2>{10, 30});
//         // REQUIRE(A.select(I(0, 10), I(0, 30), 1).shape() == std::array<int, 2>{10, 30});
//     }
// }


// TEST_CASE("ndarray<1> can be sliced, copied, and compared", "[ndarray::select]")
// {
//     auto A = ndarray<1>{0, 1, 2, 3, 4};
//     auto B = ndarray<1>{0, 1, 2, 3};
//     REQUIRE(B.container() == A.select(std::make_tuple(0, 4)).copy().container());
// }


// TEST_CASE("ndarray<2> can be sliced, copied, and compared", "[ndarray::select]")
// {
//     auto A = ndarray<2>(3, 4);
//     auto B = A.select(std::make_tuple(0, 2));

//     for (int i = 0; i < A.shape()[0]; ++i)
//     {
//         for (int j = 0; j < A.shape()[1]; ++j)
//         {
//             A(i, j) = i + j;
//         }
//     }

//     for (int i = 0; i < B.shape()[0]; ++i)
//     {
//         for (int j = 0; j < B.shape()[1]; ++j)
//         {
//             REQUIRE(B(i, j) == i + j);
//         }
//     }
// }


// TEST_CASE("ndarray<1> can be indexed const-correctly", "[ndarray]")
// {
//     const auto A = ndarray<1>{0, 1, 2, 3};
//     auto x = A(0);
//     auto B = A[0];
//     REQUIRE(x == B);
//     REQUIRE(! B.shares(A));
// }


// TEST_CASE("ndarray<2> can be sliced, indexed, and copied const-correctly", "[ndarray]")
// {
//     auto A = ndarray<2>(10, 10);

//     for (int i = 0; i < A.shape()[0]; ++i)
//     {
//         for (int j = 0; j < A.shape()[1]; ++j)
//         {
//             A(i, j) = i + j;
//         }
//     }

//     const auto Ac = A;
//     ndarray<1> B = Ac[0];
//     ndarray<2> C = Ac;
//     ndarray<2> D = A;

//     REQUIRE(Ac(0, 0) == B(0));
//     REQUIRE(Ac(0, 1) == B(1));
//     REQUIRE(! B.shares(Ac));
//     REQUIRE(! C.shares(Ac));
//     REQUIRE(D.shares(A));
//     REQUIRE(D.is(A));
// }


// TEST_CASE("ndarray<2> can be sliced and assigned to", "[ndarray]")
// {
//     auto A = ndarray<2>(10, 10);
//     auto B = ndarray<1>(10);

//     B = 1.0;
//     A[0] = B;

//     REQUIRE(B(0) == 1.0);
//     REQUIRE(A(0, 0) == 1.0);
// }


// TEST_CASE("ndarray<3> can be default-constructed and scalar-assigned properly", "[ndarray]")
// {
//     ndarray<2> A(10, 10);
//     ndarray<2> D;

//     REQUIRE(! A.empty());
//     REQUIRE(D.empty());

//     D.resize(10, 10);
//     D = 2.0;

//     REQUIRE(D.size() == A.size());
//     REQUIRE(! D.is(A));
//     REQUIRE(D(5, 5) == 2.0);

//     D.become(A);
//     REQUIRE(D.is(A));
// }


// TEST_CASE("ndarray<2> works with basic arithmetic operations", "[ndarray]")
// {
//     ndarray<2> A(10, 10);
//     A = 2.0;

//     REQUIRE((A + 1.0)(5, 5) == 3.0);
//     REQUIRE((A - 1.0)(5, 5) == 1.0);
//     REQUIRE((A * 2.0)(5, 5) == 4.0);
//     REQUIRE((A / 2.0)(5, 5) == 1.0);

//     REQUIRE((A + A)(5, 5) == 4.0);
//     REQUIRE((A - A)(5, 5) == 0.0);
//     REQUIRE((A * A)(5, 5) == 4.0);
//     REQUIRE((A / A)(5, 5) == 1.0);

//     SECTION("+= scalar works") { A += 1.0; REQUIRE(A(5, 5) == 3.0); }
//     SECTION("-= scalar works") { A -= 1.0; REQUIRE(A(5, 5) == 1.0); }
//     SECTION("*= scalar works") { A *= 2.0; REQUIRE(A(5, 5) == 4.0); }
//     SECTION("/= scalar works") { A /= 2.0; REQUIRE(A(5, 5) == 1.0); }

//     SECTION("+= ndarray works") { A += A; REQUIRE(A(5, 5) == 4.0); }
//     SECTION("-= ndarray works") { A -= A; REQUIRE(A(5, 5) == 0.0); }
//     SECTION("*= ndarray works") { A *= A; REQUIRE(A(5, 5) == 4.0); }
//     SECTION("/= ndarray works") { A /= A; REQUIRE(A(5, 5) == 1.0); }
// }


// TEST_CASE("ndarray<2> can be created by stacking 1D arrays", "[ndarray::stack]")
// {
//     auto A = ndarray<1>(100);
//     auto B = ndarray<2>::stack({A, A, A});
//     REQUIRE(B.shape() == std::array<int, 2>{3, 100});
// }
