#include <stdexcept>
#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <spectra/SymEigsSolver.h>
#include "diffusion_map.h"
#include "distance_matrix.h"

using namespace Spectra;

namespace dmaps
{
    diffusion_map::diffusion_map(const distance_matrix& dm, const vector_t& w) : 
    d_(dm.get_distances()), w_(w)
    {
        check_params();
    }

    diffusion_map::diffusion_map(const matrix_t& d, const vector_t& w) : 
    d_(d), w_(w)
    {
        check_params();        
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
        return (-0.5/eps*d_.array().square()).exp().sum();
    }

    void diffusion_map::compute(int n)
    {
        if(eps_.size() == 0)
            throw std::runtime_error("Kernel bandwidth must be defined before computing diffusion coordinates.");
        
        // Compute similarity matrix and row normalze
        // to get right stochastic matrix.
        k_.noalias() = (-0.5*d_.array().square()/(eps_*eps_.transpose()).array()).exp().matrix();
        vector_t rsum =  k_.rowwise().sum();
        k_ = rsum.asDiagonal().inverse()*k_;

        // Define eigensolver.
        DenseSymMatProd<f_type> op(k_);
        SymEigsSolver<f_type, LARGEST_ALGE, DenseSymMatProd<f_type>> eigs(&op, n, 6);

        // Solve. 
        eigs.init();
        eigs.compute();
        
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