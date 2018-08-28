#pragma once 

#include "types.h"

namespace dmaps
{
    f_type rmsd(const vector_t&, const vector_t&, const vector_t&);
    f_type euclidean(const vector_t&, const vector_t&, const vector_t&);
    f_type contact_map(const vector_t&, const vector_t&, const vector_t&);
}