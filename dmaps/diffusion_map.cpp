#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <Eigen/Core>
#include <spectra/SymEigsSolver.h>
#include "diffusion_map.h"
#include "distance_matrix.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Spectra;

namespace dmaps
{
    diffusion_map::diffusion_map(const distance_matrix& dm, const vector_t& w, int num_threads) : 
    d_(dm.get_distances()), w_(w)
    {
        check_params();
        
        #ifdef _OPENMP
        if(num_threads) omp_set_num_threads(num_threads);
        #endif
    }

    diffusion_map::diffusion_map(const matrix_t& d, const vector_t& w, int num_threads) : 
    d_(d), w_(w)
    {
        check_params();
        
        #ifdef _OPENMP
        if(num_threads) omp_set_num_threads(num_threads);
        #endif     
    }

    void diffusion_map::check_params()
    {
        if(d_.cols() != d_.rows())
            throw std::invalid_argument("Distance matrix must be square.");
        
        if(w_.size() == 0)
            w_ = vector_t::Ones(d_.cols());
        else if(w_.size() != d_.cols())
            throw std::invalid_argument("Weights vector length must match distance matrix size.");
    }

    void diffusion_map::set_kernel_bandwidth(f_type eps)
    {
        set_kernel_bandwidth(std::sqrt(eps)*vector_t::Ones(w_.size()));
    }

    void diffusion_map::set_kernel_bandwidth(const vector_t& eps)
    {
        if((eps.array() <= 0).any())
            throw std::invalid_argument("Kernel bandwidth must be positive.");
        if(eps.size() != w_.size())
            throw std::invalid_argument("Kernel bandwidth length must match distance matrix size.");
        
        eps_ = eps;
    }

    const vector_t& diffusion_map::get_kernel_bandwidth() const
    {
        return eps_;
    }

    f_type diffusion_map::sum_similarity_matrix(f_type eps) const
    {
        matrix_t wwt = w_*w_.transpose();
        return ((-0.5/eps*d_.array().square()).exp()*wwt.array()).sum();
    }

    void diffusion_map::estimate_local_scale(int k)
    {
        eps_ = vector_t::Ones(d_.rows());

        // Default choice of k.
        if(k == 0) k = static_cast<int>(std::sqrt(d_.rows()));

        // Set local epsilon scale for each entry.
        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            // Create sort indexer.
            std::vector<size_t> idx(d_.rows());
            std::iota(std::begin(idx), std::end(idx), static_cast<size_t>(0));

            #ifdef _OPENMP
            #pragma for schedule(static)
            #endif
            for(size_t i = 0; i < eps_.size(); ++i)
            {
                const vector_t& dist = d_.row(i);

                // Get indices of sorted distances (ascending).
                std::sort(std::begin(idx), std::end(idx),
                    [&](size_t a, size_t b) { return dist[a] < dist[b]; }
                );
            
                // Sum and determine weight.
                // We skip the first element which is itself.
                f_type sum = 0.;
                for(size_t j = 1; j < idx.size(); ++j)
                {
                    sum += w_[idx[j]];
                    // Break at k value.
                    if(sum >= k)
                    {
                        eps_[i] = dist[idx[j]];
                        break;
                    }
                }
            }
        }
    }

    void diffusion_map::compute(int n)
    {
        if(eps_.size() == 0)
            throw std::runtime_error("Kernel bandwidth must be defined before computing diffusion coordinates.");
        
        // Compute similarity matrix and row normalize
        // to get right stochastic matrix.
        k_ = -0.5*d_.cwiseProduct(d_);
        k_.array() /= (eps_*eps_.transpose()).array();
        k_.array() = k_.array().exp();
        k_.array() *= (w_*w_.transpose()).array();
        vector_t rsum =  k_.rowwise().sum();
        k_ = rsum.asDiagonal().inverse()*k_;

        // Define eigensolver.
        DenseSymMatProd<f_type> op(k_);
        SymEigsSolver<f_type, LARGEST_ALGE, DenseSymMatProd<f_type>> eigs(&op, n, 2*n);
        
        // Solve. 
        eigs.init();
        eigs.compute(5000, 1.e-14);
        
        if(eigs.info() != SUCCESSFUL)
            throw std::runtime_error("Eigensolver did not converge.");
        
        dvals_ = eigs.eigenvalues();
        dvecs_ = eigs.eigenvectors();
    }

    const matrix_t& diffusion_map::get_eigenvectors() const
    {
        return dvecs_;
    }

    const vector_t& diffusion_map::get_eigenvalues() const
    {
        return dvals_;
    }

    const matrixc_t& diffusion_map::get_kernel_matrix() const
    {
        return k_;
    }
}