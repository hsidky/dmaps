#include "distance_matrix.h"

namespace dmaps
{
	const matrix_t& distance_matrix::get_coordinates()
	{
		return x_;
	}

	const matrix_t& distance_matrix::get_distances()
	{
		return d_;
	}

	void distance_matrix::compute(const std::function<f_type(vector_t, vector_t, const vector_t&)>& dist)
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
}