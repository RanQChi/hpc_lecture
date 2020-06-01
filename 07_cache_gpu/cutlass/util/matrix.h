/******************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

/**
 * \file
 * Matrix data structure providing basic CPU-based algorithms and
 * operations that can be cloned and synchronized in GPU device memory
 */

#include <vector>
#include <fstream>
#include "half.h"


namespace cutlass {

/**
 * \brief Matrix data structure providing basic CPU-based algorithms and
 * operations that be synchronized with a GPU-based replica
 */
template <typename value_t>
struct matrix
{
    // Host value type (must be convertible to/from value_t)
    typedef typename nv_std::conditional<
            (nv_std::is_same<value_t, __half>::value),  // If (value_t == __half) ...
            half_t,                                     // ... use half_t internally for host storage, else...
            value_t>::type                              // ... use value_t directly
        host_value_t;


    //-----------------------------------------------------------------------------
    // Data members
    //-----------------------------------------------------------------------------

private:

    /// M dimension (height in rows)
    int _m;

    /// N dimension (width in columns)
    int _n;

    /// Data array on host
    std::vector<host_value_t> _h_data;

    /// Clone of data array on GPU device
    value_t *_d_data;

    /// GPU Device identifier that clone synchronizes with
    int _device_id;

public:

    //-----------------------------------------------------------------------------
    // Lifetime and synchronization
    //-----------------------------------------------------------------------------

    /**
     * Constructor: zero-initializes the matrix.
     */
    matrix(
        int m,  ///< Height of the matrix in rows
        int n)  ///< Width of the matrix in columns
    :
        _m(m),
        _n(n),
        _d_data(NULL),
        _device_id(0)
    {
        _h_data.resize(_m * _n, 0);
        CUDA_PERROR_EXIT(cudaMalloc((void ** )&_d_data, sizeof(value_t) * _m * _n));
        CUDA_PERROR_EXIT(cudaGetDevice(&_device_id));
    }


    /**
     * Return item at (x, y) coordinate of matrix, subject to the optional \p transform op
     */
    host_value_t get(
        int x,
        int y,
        matrix_transform_t transpose_op = matrix_transform_t::NonTranspose) const
    {
        switch (transpose_op)
        {
            case matrix_transform_t::NonTranspose :    return _h_data[y + (x * _m)];
            case matrix_transform_t::Transpose :       return _h_data[x + (y * _m)];
            default: return 0;
        }
    }

    /**
     * Get device data pointer
     */
    value_t * d_data()
    {
        return _d_data;
    }

    void fill_ramp(
        host_value_t rs,
        host_value_t cs)
    {
        for (int x = 0; x < _n; x++)
        {
            for (int y = 0; y < _m; y++)
            {
                _h_data[y + (x * _m)] = host_value_t((y * rs) + (x * cs));
            }
        }
    }

  void random() {
    for (int j = 0; j < _n; j++) {
      for (int i = 0; i < _m; i++) {
        _h_data[i + j * _m] = drand48();
      }
    }
  }



};


} // namespace cutlass
