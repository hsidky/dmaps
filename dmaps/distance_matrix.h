#pragma once 

#include <stdexcept>
#include <functional>
#include "types.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dmaps
{
    class distance_matrix
    {
    private:
        // Distance matrix.
        matrix_t d_;

        // Weights vector. 
        vector_t w_;

        // Coordinates.
        matrix_t x_;
    
    public:
        distance_matrix(const matrix_t& x, int num_threads = 1) : 
        x_(x)
        {
            #ifdef _OPENMP
            if(num_threads) omp_set_num_threads(num_threads);
            #endif

            w_ = vector_t::Ones(x_.cols()/3);
        }

        distance_matrix(const matrix_t& x, const vector_t& w, int num_threads = 1) : 
        x_(x), w_(w)
        {
            if(x_.cols()/3 != w.size())
                throw std::runtime_error("Length of weights vector must match dimension of coordinates.");

            #ifdef _OPENMP
            if(num_threads) omp_set_num_threads(num_threads);
            #endif
    
        }

        const matrix_t& get_coordinates(); 

        const matrix_t& get_distances();

        void compute(const std::function<f_type(vector_t, vector_t, const vector_t&)>&);
    };
}