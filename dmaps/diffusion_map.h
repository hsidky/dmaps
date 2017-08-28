#pragma once 

#include <string>
#include "types.h"

namespace dmaps
{
    class diffusion_map
    {
    private: 
        // Reference to distance matrix. 
        const matrix_t& d_;
        
        // Weights associated with each point.
        vector_t w_; 

        // Diffusion map eigenvectors.
        matrix_t dvecs_;

        // Diffusion map eigenvalues. 
        vector_t dvals_; 

        // Kernel matrix.
        matrix_t k_; 

        // Kernel bandwidth(s).
        vector_t eps_;

        void check_params();
    
    public:
        diffusion_map(const matrix_t& d, const vector_t& w);
        diffusion_map(const class distance_matrix& dm, const vector_t& w);
        
        void set_kernel_bandwidth(f_type eps); 
        void set_kernel_bandwidth(const vector_t& eps); 
        const vector_t& get_kernel_bandwidth() const;

        f_type sum_similarity_matrix(f_type eps) const;
        //void estimate_local_scale(int k);

        void compute(int n);

        const matrix_t& get_eigenvectors() const;
        const vector_t& get_eigenvalues() const;
        const matrix_t& get_kernel_matrix() const;        
    };
}