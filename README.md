# Introduction


This project is a header-only implementation of a numpy-like `ndarray` template for pure C++14. It should be comparable to (but smaller and more modern than) [Boost.MultiArray](https://www.boost.org/doc/libs/1_68_0/libs/multi_array/doc/index.html).


If you are interested in a featureful and professionally maintained numerical package for C++, you should check out [xtensor](https://github.com/QuantStack/xtensor).


However, if you prefer a lightweight `ndarray` container and not much else, this project might interest you.


The code should be transparent enough that you can modify it without much trouble. Pull requests welcome!


# Overview

`ndarray` objects use the same memory model as `np.array` in numpy. The array itself is a lightweight stack object containing a `std::shared_ptr` to a memory block, which may be in use by multiple arrays. Const-correctness is respected: const arrays cannot modify their memory buffers, and non-const arrays are constructed from const arrays by creating a new memory buffer.


```c++
  // Basic usage:

  nd::ndarray<int, 3> A(100, 200, 10);
  nd::ndarray<int, 2> B = A[0]; // B.shape() == {200, 10} and B.shares(A)
  nd::ndarray<int, 1> C = B[0]; // C.shape() == {10} and C.shares(B)
  nd::ndarray<int, 0> D = C[0]; // D.shape() == {} and D.shares(C)
  double d = D; // rank-0 arrays cast to underlying scalar type
  double e = A[0][0][0]; // d == e (slow)
  double f = A(0, 0, 0); // e == f (fast)
```


```c++
  // Creating a 1D array from an initializer list

  auto A = nd::ndarray<double, 1>{0, 1, 2, 3};
  A(0) = 3.0;
  A(1) = 2.0;
```


```c++
  // Multi-dimensional selections

  auto A = nd::ndarray<int, 3>(100, 200, 10);
  auto B = A.select(0, std::make_tuple(100, 150), 0); // A.rank == 1 and A.shares(B)
  
  auto _ = nd::axis::all();
  auto C = A.select(0, _|100|150, 0); // (B == C).all()
```


```c++
  // STL-compatible iteration

  auto x = 0.0;
  auto A = nd::ndarray<double, 3>(100, 200, 10);

  for (auto &a : A[50])
  {
    a = x += 1.0;
  }
  auto vector_data = std::vector<double>(A[50].begin(), A[50].end());
```


```c++
  // Respects const-correctness

  // If A is non-const, then
  {
    nd::ndarray<double, 1> A(100); // A[0].shares(A)
    A(0) = 1.0; // OK
    A.select(_|0|5) = 2.0; // OK
    nd::ndarray<double, 1> B = A; // B.is(A)
  }

  // whereas if A is const,
  {
    const nd::ndarray<double, 1> A(100); // A[0].shares(A), however
    const nd::ndarray<double, 1> B = A;  // ! B.shares(A), and
    nd::ndarray<double, 1> C = A;        // ! C.shares(A),
    // since otherwise C[0] = 1.0 would modify A's buffer and A is const.
    auto D = A[0]; // D.shares(A), however D.is_const_ref(), which has only const methods.

    // A(0) = 1.0; // compile error
    // A.select(_|0|5) = 2.0; // compile error
  }
```


```c++
  // Basic arithmetic and comparison expressions

  auto A = nd::arange<int>(10);
  auto B = nd::ones<int>(10);
  auto C = (A + B) / 2.0;
  assert(! (A == B).all());
  assert(  (A == B).any());
```


# Priority To-Do items:
- [x] Generalize scalar data type from double
- [x] Basic arithmetic operations
- [x] Allow for skips along ndarray axes
- [x] Support for comparison operators >=, <=, etc.
- [ ] Indexing via linear selections, enabling e.g. A[A > 0] = ...
- [ ] Relative indexing (negative counts backwards from end)
- [ ] Array transpose (and general axis permutation)
- [x] Factories: zeros, ones, arange
- [ ] Custom allocators (allow e.g. numpy interoperability or user memory pool)
- [x] Binary serialization
- [x] Bounds checking
- [ ] Enable/disable bounds-checking at compile time
