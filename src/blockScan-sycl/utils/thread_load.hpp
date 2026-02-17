/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * Thread utilities for reading memory using PTX cache modifiers.
 */

#pragma once

#include <iterator>
/**
 * \addtogroup UtilIo
 * @{
 */

//-----------------------------------------------------------------------------
// Tags and constants
//-----------------------------------------------------------------------------



/**
 * \brief Enumeration of cache modifiers for memory load operations.
 */
enum CacheLoadModifier
{
    LOAD_DEFAULT,       ///< Default (no modifier)
    LOAD_CA,            ///< Cache at all levels
    LOAD_CG,            ///< Cache at global level
    LOAD_CS,            ///< Cache streaming (likely to be accessed once)
    LOAD_CV,            ///< Cache as volatile (including cached system lines)
    LOAD_LDG,           ///< Cache as texture
    LOAD_VOLATILE,      ///< Volatile (any memory space)
};


/**
 * \name Thread I/O (cache modified)
 * @{
 */

/**
 * \brief Thread utility for reading memory using cub::CacheLoadModifier cache modifiers.  Can be used to load any data type.
 *
 * \par Example
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/thread/thread_load.cuh>
 *
 * // 32-bit load using cache-global modifier:
 * int *d_in;
 * int val = cub::ThreadLoad<cub::LOAD_CA>(d_in + threadIdx.x);
 *
 * // 16-bit load using default modifier
 * short *d_in;
 * short val = cub::ThreadLoad<cub::LOAD_DEFAULT>(d_in + threadIdx.x);
 *
 * // 256-bit load using cache-volatile modifier
 * double4 *d_in;
 * double4 val = cub::ThreadLoad<cub::LOAD_CV>(d_in + threadIdx.x);
 *
 * // 96-bit load using cache-streaming modifier
 * struct TestFoo { bool a; short b; };
 * TestFoo *d_struct;
 * TestFoo val = cub::ThreadLoad<cub::LOAD_CS>(d_in + threadIdx.x);
 * \endcode
 *
 * \tparam MODIFIER             <b>[inferred]</b> CacheLoadModifier enumeration
 * \tparam InputIteratorT       <b>[inferred]</b> Input iterator type \iterator
 */
template <CacheLoadModifier MODIFIER, typename InputIteratorT>
inline value_t<InputIteratorT> ThreadLoad(InputIteratorT itr);

//@}  end member group

/// Helper structure for templated load iteration (inductive case)
template <int COUNT, int MAX>
struct IterateThreadLoad
{
    template <CacheLoadModifier MODIFIER, typename T>
    static inline void Load(T const *ptr, T *vals)
    {
        vals[COUNT] = ThreadLoad<MODIFIER>(ptr + COUNT);
        IterateThreadLoad<COUNT + 1, MAX>::template Load<MODIFIER>(ptr, vals);
    }

    template <typename InputIteratorT, typename T>
    static inline void Dereference(InputIteratorT itr, T *vals)
    {
        vals[COUNT] = itr[COUNT];
        IterateThreadLoad<COUNT + 1, MAX>::Dereference(itr, vals);
    }
};


/// Helper structure for templated load iteration (termination case)
template <int MAX>
struct IterateThreadLoad<MAX, MAX>
{
    template <CacheLoadModifier MODIFIER, typename T>
    static inline void Load(T const * /*ptr*/, T * /*vals*/) {}

    template <typename InputIteratorT, typename T>
    static inline void Dereference(InputIteratorT /*itr*/,
                                            T * /*vals*/) {}
};





/**
 * ThreadLoad definition for LOAD_DEFAULT modifier on iterator types
 */
template <typename InputIteratorT>
inline value_t<InputIteratorT>
ThreadLoad(InputIteratorT itr, Int2Type<LOAD_DEFAULT> /*modifier*/,
           Int2Type<false> /*is_pointer*/)
{
    return *itr;
}


/**
 * ThreadLoad definition for LOAD_DEFAULT modifier on pointer types
 */
template <typename T>
inline T ThreadLoad(T *ptr, Int2Type<LOAD_DEFAULT> /*modifier*/,
                             Int2Type<true> /*is_pointer*/)
{
    return *ptr;
}


/**
 * ThreadLoad definition for LOAD_VOLATILE modifier on primitive pointer types
 */
template <typename T>
inline T ThreadLoadVolatilePointer(T *ptr,
                                            Int2Type<true> /*is_primitive*/)
{
    T retval = *reinterpret_cast<volatile T*>(ptr);
    return retval;
}


/**
 * ThreadLoad definition for LOAD_VOLATILE modifier on non-primitive pointer types
 */
template <typename T>
inline T ThreadLoadVolatilePointer(T *ptr,
                                            Int2Type<false> /*is_primitive*/)
{
    typedef typename UnitWord<T>::VolatileWord VolatileWord;   // Word type for memcopying

    const int VOLATILE_MULTIPLE = sizeof(T) / sizeof(VolatileWord);

    T retval;
    VolatileWord *words = reinterpret_cast<VolatileWord*>(&retval);
    IterateThreadLoad<0, VOLATILE_MULTIPLE>::Dereference(
        reinterpret_cast<volatile VolatileWord*>(ptr),
        words);
    return retval;
}


/**
 * ThreadLoad definition for LOAD_VOLATILE modifier on pointer types
 */
template <typename T>
inline T ThreadLoad(T *ptr, Int2Type<LOAD_VOLATILE> /*modifier*/,
                             Int2Type<true> /*is_pointer*/)
{
    // Apply tags for partial-specialization
    return ThreadLoadVolatilePointer(ptr, Int2Type<Traits<T>::PRIMITIVE>());
}


/**
 * ThreadLoad definition for generic modifiers on pointer types
 */
template <typename T, int MODIFIER>
inline T ThreadLoad(T const *ptr, Int2Type<MODIFIER> /*modifier*/,
                             Int2Type<true> /*is_pointer*/)
{
    typedef typename UnitWord<T>::DeviceWord DeviceWord;

    const int DEVICE_MULTIPLE = sizeof(T) / sizeof(DeviceWord);

    DeviceWord words[DEVICE_MULTIPLE];

    IterateThreadLoad<0, DEVICE_MULTIPLE>::template Load<CacheLoadModifier(MODIFIER)>(
        reinterpret_cast<DeviceWord*>(const_cast<T*>(ptr)),
        words);

    return *reinterpret_cast<T*>(words);
}


/**
 * ThreadLoad definition for generic modifiers
 */
template <CacheLoadModifier MODIFIER, typename InputIteratorT>
inline value_t<InputIteratorT> ThreadLoad(InputIteratorT itr)
{
    // Apply tags for partial-specialization
    return ThreadLoad(
        itr,
        Int2Type<MODIFIER>(),
        Int2Type<std::is_pointer<InputIteratorT>::value>());
}
