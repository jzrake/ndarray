#pragma once




// ============================================================================
namespace nd // ND_API_START
{
    template<typename T> class buffer;
} // ND_API_END




// ============================================================================
template<typename T> // ND_IMPL_START
class nd::buffer
{
public:
    using size_type = std::size_t;

    buffer() {}

    buffer(const buffer<T>& other)
    {
        count = other.count;
        memory = new T[count];

        for (int n = 0; n < count; ++n)
        {
            memory[n] = other.memory[n];
        }
    }

    buffer(buffer<T>&& other)
    {
        delete [] memory;
        memory = other.memory;
        count = other.count;

        other.memory = nullptr;
        other.count = 0;
    }

    explicit buffer(size_type count, const T& value = T()) : count(count)
    {
        memory = new T[count];

        for (int n = 0; n < count; ++n)
        {
            memory[n] = value;
        }
    }

    template< class InputIt >
    buffer(InputIt first, InputIt last)
    {
        {
            auto it = first;
            count = 0;

            while (it != last)
            {
                ++it;
                ++count;
            }
            memory = new T[count];
        }

        {
            auto it = first;
            auto n = 0;

            while (it != last)
            {
                memory[n] = *it;
                ++it;
                ++n;
            }
        }
    }

    ~buffer()
    {
        delete [] memory;
    }

    buffer<T>& operator=(const buffer<T>& other)
    {
        delete [] memory;

        count = other.count;
        memory = new T[count];

        for (int n = 0; n < count; ++n)
        {
            memory[n] = other.memory[n];
        }
        return *this;
    }

    buffer<T>& operator=(buffer<T>&& other)
    {
        delete [] memory;
        memory = other.memory;
        count = other.count;

        other.memory = nullptr;
        other.count = 0;
        return *this;
    }

    bool operator==(const buffer<T>& other) const
    {
        if (count != other.count)
        {
            return false;
        }

        for (int n = 0; n < size(); ++n)
        {
            if (memory[n] != other.memory[n])            
            {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const buffer<T>& other) const
    {
        return ! operator==(other);
    }

    size_type size() const
    {
        return count;
    }

    const T* data() const
    {
        return memory;
    }

    T* data()
    {
        return memory;
    }

    const T& operator[](size_type offset) const
    {
        return memory[offset];
    }

    T& operator[](size_type offset)
    {
        return memory[offset];
    }

    T* begin() { return memory; }
    T* end() { return memory + count; }

    const T* begin() const { return memory; }
    const T* end() const { return memory + count; }

private:
    T* memory = nullptr;
    size_type count = 0;
}; // ND_IMPL_END




// ============================================================================
#ifdef TEST_BUFFER
#include "catch.hpp"


TEST_CASE("buffer can be constructed from", "[buffer]")
{
    SECTION("Can instantiate an empty buffer")
    {
        nd::buffer<double> B;
        REQUIRE(B.size() == 0);
        REQUIRE(B.data() == nullptr);
    }

    SECTION("Can instantiate a constant buffer")
    {
        nd::buffer<double> B(100, 1.5);
        REQUIRE(B.size() == 100);
        REQUIRE(B.data() != nullptr);
        REQUIRE(B[0] == 1.5);
        REQUIRE(B[99] == 1.5);
    }

    SECTION("Can instantiate a buffer from input iterator")
    {
        std::vector<int> A{0, 1, 2, 3};
        buffer<double> B(A.begin(), A.end());
        REQUIRE(B.size() == 4);
        REQUIRE(B[0] == 0);
        REQUIRE(B[1] == 1);
        REQUIRE(B[2] == 2);
        REQUIRE(B[3] == 3);
    }

    SECTION("Can move-construct and move-assign a buffer")
    {
        nd::buffer<double> A(100, 1.5);
        nd::buffer<double> B(200, 2.0);

        B = std::move(A);

        REQUIRE(A.size() == 0);
        REQUIRE(A.data() == nullptr);

        REQUIRE(B.size() == 100);
        REQUIRE(B[0] == 1.5);
        REQUIRE(B[99] == 1.5);

        auto C = std::move(B);

        REQUIRE(B.size() == 0);
        REQUIRE(B.data() == nullptr);
        REQUIRE(C.size() == 100);
        REQUIRE(C[0] == 1.5);
        REQUIRE(C[99] == 1.5);
    }

    SECTION("Equality operators between buffers work correctly")
    {
        nd::buffer<double> A(100, 1.5);   
        nd::buffer<double> B(100, 1.5);
        nd::buffer<double> C(200, 1.5);
        nd::buffer<double> D(100, 2.0);

        REQUIRE(A == A);
        REQUIRE(A == B);
        REQUIRE(A != C);
        REQUIRE(A != D);

        REQUIRE(B == A);
        REQUIRE(B == B);
        REQUIRE(B != C);
        REQUIRE(B != D);

        REQUIRE(C != A);
        REQUIRE(C != B);
        REQUIRE(C == C);
        REQUIRE(C != D);

        REQUIRE(D != A);
        REQUIRE(D != B);
        REQUIRE(D != C);
        REQUIRE(D == D);
    }
}


#endif // TEST_BUFFER