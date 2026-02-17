#pragma once

#include <sycl/sycl.hpp>
#include <cfloat>

#define LOG_WARP_THREADS(unused) (5)
#define WARP_THREADS(unused) (1 << LOG_WARP_THREADS(0))
#define PTX_WARP_THREADS        WARP_THREADS(0)
#define PTX_LOG_WARP_THREADS    LOG_WARP_THREADS(0)

// not used in warpscan
#define LOG_SMEM_BANKS(unused) (5)
#define SMEM_BANKS(unused) (1 << LOG_SMEM_BANKS(0))

template <typename NumeratorT, typename DenominatorT>
inline constexpr NumeratorT DivideAndRoundUp(NumeratorT n,
                                                      DenominatorT d)
{
  return static_cast<NumeratorT>(n / d + (n % d != 0 ? 1 : 0));
}

/**
 * \brief Statically determine if N is a power-of-two
 */
template <int N>
struct PowerOfTwo
{
    enum { VALUE = ((N & (N - 1)) == 0) };
};

struct Sum
{
  /// Binary sum operator, returns `t + u`
  template <typename T, typename U>
  inline auto operator()(T &&t,
                                  U &&u) const -> decltype(std::forward<T>(t) +
                                                           std::forward<U>(u))
  {
    return std::forward<T>(t) + std::forward<U>(u);
  }
};

#define MAX(a, b) (((b) > (a)) ? (b) : (a))
#define MIN(a, b) (((b) < (a)) ? (b) : (a))

struct Max
{
  /// Boolean max operator, returns `(t > u) ? t : u`
  template <typename T, typename U>
  inline typename std::common_type<T, U>::type operator()(T &&t,
                                                                   U &&u) const
  {
    return MAX(t, u);
  }
};

struct Min
{
  /// Boolean max operator, returns `(t > u) ? t : u`
  template <typename T, typename U>
  inline typename std::common_type<T, U>::type operator()(T &&t,
                                                                   U &&u) const
  {
    return MIN(t, u);
  }
};


template <int A>
struct Int2Type
{
    enum {VALUE = A};
};

/**
 * \brief Statically determine log2(N), rounded up.
 *
 * For example:
 *     Log2<8>::VALUE   // 3
 *     Log2<3>::VALUE   // 2
 */
template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2
{
    /// Static logarithm value
    enum { VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE };         // Inductive case
};

template <int N, int COUNT>
struct Log2<N, 0, COUNT>
{
    enum {VALUE = (1 << (COUNT - 1) < N) ?                                  // Base case
        COUNT :
        COUNT - 1 };
};

struct NullType
{
    using value_type = NullType;

    template <typename T>
    inline NullType &operator=(const T &) {
        return *this;
    }

    inline bool operator==(const NullType &) { return true; }

    inline bool operator!=(const NullType &) { return false; }
};


/// Structure alignment
template <typename T>
struct AlignBytes
{
    struct Pad
    {
        T       val;
        char    byte;
    };

    enum
    {
        /// The "true CUDA" alignment of T in bytes
        ALIGN_BYTES = sizeof(Pad) - sizeof(T)
    };

    /// The "truly aligned" type
    typedef T Type;
};

#define __ALIGN_BYTES(t, b)                                                    \
template <> struct AlignBytes<t>                                               \
{ enum { ALIGN_BYTES = b }; typedef __attribute__((aligned(b))) t Type; };

__ALIGN_BYTES(sycl::short4, 8)
__ALIGN_BYTES(sycl::ushort4, 8)
__ALIGN_BYTES(sycl::int2, 8)
__ALIGN_BYTES(sycl::uint2, 8)
__ALIGN_BYTES(long long, 8)
__ALIGN_BYTES(unsigned long long, 8)
__ALIGN_BYTES(sycl::float2, 8)
__ALIGN_BYTES(double, 8)
__ALIGN_BYTES(sycl::long2, 16)
__ALIGN_BYTES(sycl::ulong2, 16)
__ALIGN_BYTES(sycl::int4, 16)
__ALIGN_BYTES(sycl::uint4, 16)
__ALIGN_BYTES(sycl::float4, 16)
__ALIGN_BYTES(sycl::long4, 16)
__ALIGN_BYTES(sycl::ulong4, 16)
__ALIGN_BYTES(sycl::double2, 16)
__ALIGN_BYTES(sycl::double4, 16)

template <typename T> struct AlignBytes<volatile T> : AlignBytes<T> {};
template <typename T> struct AlignBytes<const T> : AlignBytes<T> {};
template <typename T> struct AlignBytes<const volatile T> : AlignBytes<T> {};

template <bool Test, class T1, class T2>
using conditional_t = typename std::conditional<Test, T1, T2>::type;

template <typename Iterator>
using value_t = typename std::iterator_traits<Iterator>::value_type;

/// Unit-words of data movement
template <typename T>
struct UnitWord
{
    enum {
        ALIGN_BYTES = AlignBytes<T>::ALIGN_BYTES
    };

    template <typename Unit>
    struct IsMultiple
    {
        enum {
            UNIT_ALIGN_BYTES    = AlignBytes<Unit>::ALIGN_BYTES,
            IS_MULTIPLE         = (sizeof(T) % sizeof(Unit) == 0) && (int(ALIGN_BYTES) % int(UNIT_ALIGN_BYTES) == 0)
        };
    };

    /// Biggest shuffle word that T is a whole multiple of and is not larger than
    /// the alignment of T
    using ShuffleWord = conditional_t<
      IsMultiple<int>::IS_MULTIPLE,
      unsigned int,
      conditional_t<IsMultiple<short>::IS_MULTIPLE,
                                 unsigned short,
                                 unsigned char>>;

    /// Biggest volatile word that T is a whole multiple of and is not larger than
    /// the alignment of T
    using VolatileWord =
      conditional_t<IsMultiple<long long>::IS_MULTIPLE,
                                 unsigned long long,
                                 ShuffleWord>;

    /// Biggest memory-access word that T is a whole multiple of and is not larger
    /// than the alignment of T
    using DeviceWord = conditional_t<IsMultiple<sycl::long2>::IS_MULTIPLE,
                                     sycl::ulong2, VolatileWord>;
};

template <typename T>
struct Uninitialized
{
    /// Biggest memory-access word that T is a whole multiple of and is not larger than the alignment of T
    typedef typename UnitWord<T>::DeviceWord DeviceWord;

    static constexpr std::size_t DATA_SIZE = sizeof(T);
    static constexpr std::size_t WORD_SIZE = sizeof(DeviceWord);
    static constexpr std::size_t WORDS = DATA_SIZE / WORD_SIZE;

    /// Backing storage
    DeviceWord storage[WORDS];

    /// Alias
    inline T &Alias()
    {
        return reinterpret_cast<T&>(*this);
    }
};

/**
 * \brief Returns the row-major linear thread identifier for a multidimensional thread block
 */
inline int RowMajorTid(int block_dim_x, int block_dim_y, int block_dim_z,
                       const sycl::nd_item<3> &item)
{
    return ((block_dim_z == 1)
                ? 0
                : (item.get_local_id(0) * block_dim_x * block_dim_y)) +
           ((block_dim_y == 1) ? 0 : (item.get_local_id(1) * block_dim_x)) +
           item.get_local_id(2);
}

/******************************************************************************
 * Simple type traits utilities.
 *
 * For example:
 *     Traits<int>::CATEGORY             // SIGNED_INTEGER
 *     Traits<NullType>::NULL_TYPE       // true
 *     Traits<uint4>::CATEGORY           // NOT_A_NUMBER
 *     Traits<uint4>::PRIMITIVE;         // false
 *
 ******************************************************************************/

/**
 * \brief Basic type traits categories
 */
enum Category
{
    NOT_A_NUMBER,
    SIGNED_INTEGER,
    UNSIGNED_INTEGER,
    FLOATING_POINT
};


/**
 * \brief Basic type traits
 */
template <Category _CATEGORY, bool _PRIMITIVE, bool _NULL_TYPE, typename _UnsignedBits, typename T>
struct BaseTraits
{
    /// Category
    static const Category CATEGORY      = _CATEGORY;
    enum
    {
        PRIMITIVE       = _PRIMITIVE,
        NULL_TYPE       = _NULL_TYPE,
    };
};


/**
 * Basic type traits (unsigned primitive specialization)
 */
template <typename _UnsignedBits, typename T>
struct BaseTraits<UNSIGNED_INTEGER, true, false, _UnsignedBits, T>
{
    typedef _UnsignedBits       UnsignedBits;

    static const Category       CATEGORY    = UNSIGNED_INTEGER;
    static const UnsignedBits   LOWEST_KEY  = UnsignedBits(0);
    static const UnsignedBits   MAX_KEY     = UnsignedBits(-1);

    enum
    {
        PRIMITIVE       = true,
        NULL_TYPE       = false,
    };

    static inline UnsignedBits TwiddleIn(UnsignedBits key)
    {
        return key;
    }

    static inline UnsignedBits TwiddleOut(UnsignedBits key)
    {
        return key;
    }

    static inline T Max()
    {
        UnsignedBits retval_bits = MAX_KEY;
        T retval;
        memcpy(&retval, &retval_bits, sizeof(T));
        return retval;
    }

    static inline T Lowest()
    {
        UnsignedBits retval_bits = LOWEST_KEY;
        T retval;
        memcpy(&retval, &retval_bits, sizeof(T));
        return retval;
    }
};


/**
 * Basic type traits (signed primitive specialization)
 */
template <typename _UnsignedBits, typename T>
struct BaseTraits<SIGNED_INTEGER, true, false, _UnsignedBits, T>
{
    typedef _UnsignedBits       UnsignedBits;

    static const Category       CATEGORY    = SIGNED_INTEGER;
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    static const UnsignedBits   LOWEST_KEY  = HIGH_BIT;
    static const UnsignedBits   MAX_KEY     = UnsignedBits(-1) ^ HIGH_BIT;

    enum
    {
        PRIMITIVE       = true,
        NULL_TYPE       = false,
    };

    static inline UnsignedBits TwiddleIn(UnsignedBits key)
    {
        return key ^ HIGH_BIT;
    };

    static inline UnsignedBits TwiddleOut(UnsignedBits key)
    {
        return key ^ HIGH_BIT;
    };

    static inline T Max()
    {
        UnsignedBits retval = MAX_KEY;
        return reinterpret_cast<T&>(retval);
    }

    static inline T Lowest()
    {
        UnsignedBits retval = LOWEST_KEY;
        return reinterpret_cast<T&>(retval);
    }
};

template <typename _T>
struct FpLimits;

template <>
struct FpLimits<float>
{
    static inline float Max() {
        return FLT_MAX;
    }

    static inline float Lowest() {
        return FLT_MAX * float(-1);
    }
};

template <>
struct FpLimits<double>
{
    static inline double Max() {
        return DBL_MAX;
    }

    static inline double Lowest() {
        return DBL_MAX  * double(-1);
    }
};

template <> struct FpLimits<sycl::half>
{
    static inline sycl::half Max() {
        unsigned short max_word = 0x7BFF;
        return reinterpret_cast<sycl::half &>(max_word);
    }

    static inline sycl::half Lowest() {
        unsigned short lowest_word = 0xFBFF;
        return reinterpret_cast<sycl::half &>(lowest_word);
    }
};

template <> struct FpLimits<sycl::ext::oneapi::bfloat16>
{
    static inline sycl::ext::oneapi::bfloat16 Max() {
        unsigned short max_word = 0x7F7F;
        return reinterpret_cast<sycl::ext::oneapi::bfloat16 &>(max_word);
    }

    static inline sycl::ext::oneapi::bfloat16 Lowest() {
        unsigned short lowest_word = 0xFF7F;
        return reinterpret_cast<sycl::ext::oneapi::bfloat16 &>(lowest_word);
    }
};

/**
 * Basic type traits (fp primitive specialization)
 */
template <typename _UnsignedBits, typename T>
struct BaseTraits<FLOATING_POINT, true, false, _UnsignedBits, T>
{
    typedef _UnsignedBits       UnsignedBits;

    static const Category       CATEGORY    = FLOATING_POINT;
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    static const UnsignedBits   LOWEST_KEY  = UnsignedBits(-1);
    static const UnsignedBits   MAX_KEY     = UnsignedBits(-1) ^ HIGH_BIT;

    enum
    {
        PRIMITIVE       = true,
        NULL_TYPE       = false,
    };

    static inline UnsignedBits TwiddleIn(UnsignedBits key)
    {
        UnsignedBits mask = (key & HIGH_BIT) ? UnsignedBits(-1) : HIGH_BIT;
        return key ^ mask;
    };

    static inline UnsignedBits TwiddleOut(UnsignedBits key)
    {
        UnsignedBits mask = (key & HIGH_BIT) ? HIGH_BIT : UnsignedBits(-1);
        return key ^ mask;
    };

    static inline T Max() {
        return FpLimits<T>::Max();
    }

    static inline T Lowest() {
        return FpLimits<T>::Lowest();
    }
};


/**
 * \brief Numeric type traits
 */
// clang-format off
template <typename T> struct NumericTraits :            BaseTraits<NOT_A_NUMBER, false, false, T, T> {};

template <> struct NumericTraits<NullType> :            BaseTraits<NOT_A_NUMBER, false, true, NullType, NullType> {};

template <> struct NumericTraits<char> :                BaseTraits<(std::numeric_limits<char>::is_signed) ? SIGNED_INTEGER : UNSIGNED_INTEGER, true, false, unsigned char, char> {};
template <> struct NumericTraits<signed char> :         BaseTraits<SIGNED_INTEGER, true, false, unsigned char, signed char> {};
template <> struct NumericTraits<short> :               BaseTraits<SIGNED_INTEGER, true, false, unsigned short, short> {};
template <> struct NumericTraits<int> :                 BaseTraits<SIGNED_INTEGER, true, false, unsigned int, int> {};
template <> struct NumericTraits<long> :                BaseTraits<SIGNED_INTEGER, true, false, unsigned long, long> {};
template <> struct NumericTraits<long long> :           BaseTraits<SIGNED_INTEGER, true, false, unsigned long long, long long> {};

template <> struct NumericTraits<unsigned char> :       BaseTraits<UNSIGNED_INTEGER, true, false, unsigned char, unsigned char> {};
template <> struct NumericTraits<unsigned short> :      BaseTraits<UNSIGNED_INTEGER, true, false, unsigned short, unsigned short> {};
template <> struct NumericTraits<unsigned int> :        BaseTraits<UNSIGNED_INTEGER, true, false, unsigned int, unsigned int> {};
template <> struct NumericTraits<unsigned long> :       BaseTraits<UNSIGNED_INTEGER, true, false, unsigned long, unsigned long> {};
template <> struct NumericTraits<unsigned long long> :  BaseTraits<UNSIGNED_INTEGER, true, false, unsigned long long, unsigned long long> {};
template <> struct NumericTraits<float> :               BaseTraits<FLOATING_POINT, true, false, unsigned int, float> {};
template <> struct NumericTraits<double> :              BaseTraits<FLOATING_POINT, true, false, unsigned long long, double> {};
template <> struct NumericTraits<sycl::half> :              BaseTraits<FLOATING_POINT, true, false, unsigned short, sycl::half> {};
template <> struct NumericTraits<sycl::ext::oneapi::bfloat16> :       BaseTraits<FLOATING_POINT, true, false, unsigned short, sycl::ext::oneapi::bfloat16> {};
template <> struct NumericTraits<bool> :                BaseTraits<UNSIGNED_INTEGER, true, false, typename UnitWord<bool>::VolatileWord, bool> {};
// clang-format on

/**
 * \brief Type traits
 */
template <typename T>
struct Traits : NumericTraits<typename std::remove_cv<T>::type> {};

template <int LOGICAL_WARP_THREADS>
inline unsigned int WarpMask(unsigned int warp_id)
{
  constexpr bool is_pow_of_two = PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE;
  constexpr bool is_arch_warp  = LOGICAL_WARP_THREADS == WARP_THREADS(0);

  unsigned int member_mask = 0xFFFFFFFFu >>
                             (WARP_THREADS(0) - LOGICAL_WARP_THREADS);

  if (is_pow_of_two && !is_arch_warp)
  {
    member_mask <<= warp_id * LOGICAL_WARP_THREADS;
  }

  return member_mask;
}

inline unsigned int LaneId(const sycl::nd_item<3> &item)
{
    return item.get_local_id(2) % 32;
}

inline void WARP_SYNC(unsigned int member_mask, const sycl::nd_item<3> &item)
{
    sycl::group_barrier(item.get_sub_group());
}

inline int WARP_BALLOT(int predicate, unsigned int member_mask,
                                const sycl::nd_item<3> &item)
{
    auto sg = item.get_sub_group(); 
    return sycl::reduce_over_group(
        sg,
        member_mask & (predicate ? 0x1 << sg.get_local_linear_id() : 0),
        sycl::ext::oneapi::plus<>());
}

inline unsigned int LaneMaskGt(const sycl::nd_item<3> &item)
{
   return 0xFFFFFFFF << (1 + item.get_local_id(2));
}

/**
 * \brief Returns the warp lane mask of all lanes greater than or equal to the calling thread
 */
inline unsigned int LaneMaskGe(const sycl::nd_item<3> &item)
{
   return 0xFFFFFFFF << item.get_local_id(2);
}

template <
    int NOMINAL_4B_BLOCK_THREADS,
    int NOMINAL_4B_ITEMS_PER_THREAD,
    typename T>
struct MemBoundScaling
{
    enum {
        ITEMS_PER_THREAD    = MAX(1, MIN(NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T), NOMINAL_4B_ITEMS_PER_THREAD * 2)),
        BLOCK_THREADS       = MIN(NOMINAL_4B_BLOCK_THREADS, (((1024 * 48) / (sizeof(T) * ITEMS_PER_THREAD)) + 31) / 32 * 32),
    };
};

