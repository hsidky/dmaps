#include <iostream>
#include <fstream>
#include <stdexcept>
#include "distance_matrix.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dmaps
{
    distance_matrix::distance_matrix(const matrix_t& x, int num_threads) : 
    x_(x)
    {
        #ifdef _OPENMP
        if(num_threads) omp_set_num_threads(num_threads);
        #endif

        w_ = vector_t::Ones(x.cols());
    }

    distance_matrix::distance_matrix(const matrix_t& x, const vector_t& w, int num_threads) : 
    x_(x), w_(w)
    {
        #ifdef _OPENMP
        if(num_threads) omp_set_num_threads(num_threads);
        #endif

        if(w.size() == 0) w_ = vector_t::Ones(x.cols());
    }

    // Based on https://stackoverflow.com/questions/25389480    
    distance_matrix::distance_matrix(const std::string& filename, int num_threads)
    {
        #ifdef _OPENMP
        if(num_threads) omp_set_num_threads(num_threads);
        #endif

        std::ifstream in(filename, std::ios::in | std::ios::binary);
        
        // Read in distance matrix. 
        {
            matrix_t::Index rows = 0, cols = 0;
            in.read((char*) (&rows), sizeof(matrix_t::Index));
            in.read((char*) (&cols), sizeof(matrix_t::Index));
            d_.resize(rows, cols);
            in.read((char*) d_.data(), d_.size()*sizeof(matrix_t::Scalar));
        }
        
        // Read in weights vector. 
        {
            matrix_t::Index rows = 0, cols = 0;
            in.read((char*) (&rows), sizeof(matrix_t::Index));
            in.read((char*) (&cols), sizeof(matrix_t::Index));
            w_.resize(rows, cols);
            in.read((char*) w_.data(), w_.size()*sizeof(matrix_t::Scalar));
        }
        
        // Read in coordinates matrix. 
        {
            matrix_t::Index rows = 0, cols = 0;
            in.read((char*) (&rows), sizeof(matrix_t::Index));
            in.read((char*) (&cols), sizeof(matrix_t::Index));
            x_.resize(rows, cols);
            in.read((char*) x_.data(), x_.size()*sizeof(matrix_t::Scalar));
        }
        
        in.close();
    }
    
	const matrix_t& distance_matrix::get_coordinates() const
	{
		return x_;
	}

	const matrix_t& distance_matrix::get_distances() const
	{
		return d_;
	}

	void distance_matrix::compute(const std::function<f_type(const vector_t&, const vector_t&, const vector_t&)>& dist)
	{
        int n = x_.rows();
        int m = n/2 - 1 + n%2;
        d_ = matrix_t::Zero(n, n);
        
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < n; ++i)
        {
			for(int j = 0; j < m; ++j)
			{
                int ii = i, jj = j + 1;
                if(j < i)
                {
                    ii = n - 1 - i; 
                    jj = n - 1 - j;
                }

				d_(ii,jj) = dist(x_.row(ii), x_.row(jj), w_);
				d_(jj,ii) = d_(ii,jj);
            }
        }

        if(n % 2 == 0)
        {
            #pragma omp parallel for schedule(static)        
            for(int i = 0; i < n/2; ++i)
            {
                d_(i,n/2) = dist(x_.row(i), x_.row(n/2), w_);
                d_(n/2,i) = d_(i,n/2);
            }
        }
        #else
        for(int i = 0; i < n - 1; ++i)
            for(int j = i + 1; j < n; ++j)
            {
                d_(i,j) = dist(x_.row(i), x_.row(j), w_);
                d_(j,i) = d_(i,j);
            }
        #endif
    }
    
    // Based on https://stackoverflow.com/questions/25389480
    void distance_matrix::save(const std::string& filename) const
    {
        std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
        
        // Write out distance matrix. 
        {
            matrix_t::Index rows = d_.rows(), cols = d_.cols();
            out.write((char*) (&rows), sizeof(matrix_t::Index));
            out.write((char*) (&cols), sizeof(matrix_t::Index));
            out.write((char*) d_.data(), d_.size()*sizeof(matrix_t::Scalar));
        }
        
        // Write out weights vector.
        {
            vector_t::Index rows = w_.rows(), cols = w_.cols(); 
            out.write((char*) (&rows), sizeof(vector_t::Index));
            out.write((char*) (&cols), sizeof(vector_t::Index));
            out.write((char*) w_.data(), w_.size()*sizeof(vector_t::Scalar));
        }
        
        // Write out coordinates matrix.
        {
            matrix_t::Index rows = x_.rows(), cols = x_.cols(); 
            out.write((char*) (&rows), sizeof(matrix_t::Index));
            out.write((char*) (&cols), sizeof(matrix_t::Index));
            out.write((char*) x_.data(), x_.size()*sizeof(matrix_t::Scalar));
        }

        out.close();
    }
}