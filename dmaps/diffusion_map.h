#pragma once 

#include <string>
#include "types.h"

namespace dmaps
{
    class diffusion_map
    {
    private:
        // Internal distance matrix in case copy is desired. 
        const matrix_t dint_; 
        
        // Reference to distance matrix. 
        const matrix_t& d_;
        
        // Weights (square root) associated with each point.
        vector_t w_; 

        // Diffusion map eigenvectors.
        matrix_t dvecs_;

        // Diffusion map eigenvalues. 
        vector_t dvals_; 

        // Kernel matrix.
        matrixc_t k_; 

        // Kernel bandwidth(s).
        f_type eps_;

        void check_params();
    
    public:
        diffusion_map(const matrix_t& d, const vector_t& w, int num_threads = 0);
        diffusion_map(const class distance_matrix& dm, const vector_t& w, int num_threads = 0);
        
        void set_kernel_bandwidth(f_type eps); 
        f_type get_kernel_bandwidth() const;

        f_type sum_similarity_matrix(f_type eps, f_type alpha) const;

        vector_t nystrom(const vector_t& distances, f_type alpha, f_type beta);

        void compute(int n, f_type alpha, f_type beta);

        const matrix_t& get_eigenvectors() const;
        const vector_t& get_eigenvalues() const;
        const matrixc_t& get_kernel_matrix() const;        
    };
}