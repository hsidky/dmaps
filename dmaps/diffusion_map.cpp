#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <Eigen/Core>
#include <spectra/GenEigsSolver.h>
#include "diffusion_map.h"
#include "distance_matrix.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Spectra;

namespace dmaps
{

    diffusion_map::diffusion_map(const matrix_t& d, const vector_t& w, int num_threads) : 
    dint_(d), d_(dint_), w_(w)
    {
        check_params();
        
        #ifdef _OPENMP
        if(num_threads) omp_set_num_threads(num_threads);
        #endif     
    }

    diffusion_map::diffusion_map(const distance_matrix& dm, const vector_t& w, int num_threads) : 
    d_(dm.get_distances()), w_(w)
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
        if(eps <= 0)
            throw std::invalid_argument("Kernel bandwidth must be positive."); 
        eps_ = eps; 
    }

    f_type diffusion_map::get_kernel_bandwidth() const
    {
        return eps_;
    }

    f_type diffusion_map::sum_similarity_matrix(f_type eps, f_type alpha) const
    {
        matrix_t wwt = w_*w_.transpose();
        return ((-0.5/eps*d_.array().square().pow(alpha)).exp()*wwt.array()).sum();
    }

    /*
    void diffusion_map::estimate_local_scale(int k)
    {
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
    */

    void diffusion_map::compute(int n, f_type alpha, f_type beta)
    {
        if(eps_ == 0)
            throw std::runtime_error("Kernel bandwidth must be defined before computing diffusion coordinates.");
        
        if(alpha <= 0 || alpha > 1)
            throw std::invalid_argument("Distance scaling must be in the interval (0,1].");
        
        // Compute similarity matrix and row normalize
        // to get right stochastic matrix.
        k_ = -0.5/eps_*d_.array().square().pow(alpha);
        k_.array() = k_.array().exp();
        k_.array() *= (w_*w_.transpose()).array();

        // Density normalization.
        vector_t rsum =  k_.rowwise().sum().array().pow(-beta);
        k_ = rsum.asDiagonal()*k_*rsum.asDiagonal();

        // Right stochastic matrix. 
        rsum =  k_.rowwise().sum().array().cwiseInverse();
        k_ = rsum.asDiagonal()*k_;

        // Define eigensolver.
        DenseGenMatProd<f_type> op(k_);
        GenEigsSolver <f_type, LARGEST_MAGN, DenseGenMatProd<f_type>> eigs(&op, n, 2*n);
        
        // Solve. 
        eigs.init();
        eigs.compute();
        
        if(eigs.info() != SUCCESSFUL)
            throw std::runtime_error("Eigensolver did not converge.");
        
        dvals_ = eigs.eigenvalues().real();
        dvecs_ = eigs.eigenvectors().real();
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