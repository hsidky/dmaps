#pragma once 

#include "types.h"

namespace dmaps
{
    f_type rmsd(const vector_t& ri, const vector_t& rj, const vector_t& w);

    f_type euclidean(const vector_t& ri, const vector_t& rj, const vector_t& w);
}