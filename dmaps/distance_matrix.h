#pragma once 

#include <stdexcept>
#include <functional>
#include <string>
#include "types.h"

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
        distance_matrix(const matrix_t& x, int num_threads = 1);

        distance_matrix(const matrix_t& x, const vector_t& w, int num_threads = 1);

        distance_matrix(const std::string& filename, int num_threads = 1);

        const matrix_t& get_coordinates(); 

        const matrix_t& get_distances();

        void compute(const std::function<f_type(vector_t, vector_t, const vector_t&)>&);

        void save(const std::string& filename);
    };
}