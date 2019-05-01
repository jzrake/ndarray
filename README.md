# Introduction


This is a header-only implementation of an N-dimensional array type for modern C++. It allows you to do many different types of transformations on topologically cartesian data.

This library adopts an abstract notion of an array. An array is any mapping from a space of N-dimensional indexes `(i, j, ...)` to some type of value, and which defines a rectangular region (the shape) containing the valid indexes:

    `array: (index => value, shape)`

This mapping is formed from the composition of two functions: an `access_pattern` and a `provider`. The provider is the source of data, defining a mapping from indexes to values, and a shape. The access pattern transforms and restricts the provider's index space.
