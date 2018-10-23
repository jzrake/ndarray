# ndarray



This project is an experimental, header-only implementation of a numpy-inspired ndarray template for pure C++0x. It should be comparable to (but perhaps cleaner and smaller than) Boost.MultiArray.



```C++
  // Basic usage:

  ndarray<3> A(100, 200, 10);
  ndarray<2> B = A[0]; // B.shape() == {200, 10}; B.shares(A);
  ndarray<1> C = B[0]; // C.shape() == {10}; C.shares(B);
  ndarray<0> D = C[0]; // D.shape() == {}; D.shares(C);
  double d = D; // rank-0 arrays cast to underlying scalar type
  double e = A[0][0][0]; // d == e
  double f = A(0, 0, 0); // e == f
```

```C++
  // Creating a 1D array from an initializer list

  auto A = ndarray<1>{0, 1, 2, 3};
  A(0) = 3.0;
  A(1) = 2.0;
```

```C++
  // Multi-dimensional selections

  auto A = ndarray<3>(100, 200, 10);
  auto B = A.select(0, std::make_tuple(100, 150), 0); // A.rank == 1; A.shares(B);
```

```C++
  // STL-compatible iteration

  auto x = 0.0;
  auto A = ndarray<3>(100, 200, 10);

  for (auto &a : A[50])
  {
    a = x += 1.0;
  }
  auto vector_data = std::vector<double>(A[50].begin(), A[50].end());
```
