export SparseHDF5Matrix
import HDF5
import SparseArrays

"""
Wrapper class for a HDF5 sparse array.
"""
struct SparseHDF5Matrix{T} <: HDF5Array{T, 2}
    file::String 
    name::String
    dims::Tuple{Integer,Integer}
    ptrs::Vector{Integer}

    # Mutable cache. Technically we need to protect this from multiple threads,
    # but whatever, it's not like we can do parallel HDF5 access anyway.
    cache::Dict{Integer,SparseArrays.SparseVector{Integer,T}}
end

"""
    SparseHDF5Array(file, name)

Create a `HDF5Array` for a dense dataset inside a HDF5 file `file` at name `name`.

# Examples
```jldoctest
julia> x = SparseHDF5Array("zeisel-brain/tenx.h5", "matrix")
```
"""
function SparseHDF5Matrix(file::String, name::String)
    handle = HDF5.h5open(file)
    group = handle[name]

    if !isa(group, HDF5.Group)
        throw(ErrorException("'" * name * "' in '" * file * "' should be a group"))
    end

    dims = HDF5.read(group["shape"])
    type = HDF5.get_jl_type(group["data"])
    ptrs = HDF5.read(group["indptr"])
    return SparseHDF5Matrix{type}(file, name, (dims...,), ptrs, Dict{Integer,SparseArrays.SparseVector{Integer,type}}())
end

function Base.size(x::SparseHDF5Matrix{T}) where T
    return x.dims
end

function Base.getindex(x::SparseHDF5Matrix{T}, i::Integer, j::Integer) where T
    if !haskey(x.cache, j)
        handle = HDF5.h5open(x.file)
        iset = handle[x.name * "/indices"]
        dset = handle[x.name * "/data"]

        start = Int(x.ptrs[j]) + 1 # 1-based indexing
        finish = Int(x.ptrs[j + 1]) # technically needs a -1, but this is cancelled by the +1 above

        dvals = dset[start:finish]
        ivals = iset[start:finish]
        vec = SparseArrays.sparsevec(ivals, dvals, x.dims[1])
        x.cache[j] = vec
    end

    return x.cache[j][i]
end
