export SparseHDF5Matrix
export sparse
import HDF5
import SparseArrays

"""
Wrapper class for a HDF5 sparse array.
"""
struct SparseHDF5Matrix{Tv,Ti} <: HDF5Array{Tv, 2}
    file::String 
    name::String
    dims::Tuple{Integer,Integer}
    ptrs::Vector{Integer}

    # Mutable cache. Technically we need to protect this from multiple threads,
    # but whatever, it's not like we can do parallel HDF5 access anyway.
    cache::Dict{Integer,SparseArrays.SparseVector{Ti,Tv}}
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
    dtype = HDF5.get_jl_type(group["data"])
    itype = HDF5.get_jl_type(group["indices"])
    ptrs = HDF5.read(group["indptr"])

    return SparseHDF5Matrix{dtype,itype}(file, name, (dims...,), ptrs, Dict{Integer,SparseArrays.SparseVector{itype,dtype}}())
end

function Base.size(x::SparseHDF5Matrix{Tv,Ti}) where {Tv,Ti}
    return x.dims
end

function Base.getindex(x::SparseHDF5Matrix{Tv,Ti}, i::Integer, j::Integer) where {Tv,Ti}
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

"""
    sparse(x)

Convert a `SparseHDF5Matrix` into an in-memory `SparseMatrix` of the relevant type.
"""
function sparse(x::SparseHDF5Matrix{Tv,Ti}) where {Tv,Ti}
    handle = HDF5.h5open(x.file)
    iset = handle[x.name * "/indices"]
    dset = handle[x.name * "/data"]

    values = Vector{Vector{Tv}}(undef, x.dims[2])
    indices = Vector{Vector{Int}}(undef, x.dims[2]) # Hack, we can't store the actual index here.

    # Extract by column.
    for i in 1:x.dims[2]
        start = Int(x.ptrs[i]) + 1 # 1-based indexing
        finish = Int(x.ptrs[i + 1]) # technically needs a -1, but this is cancelled by the +1 above
        values[i] = dset[start:finish]
        indices[i] = convert(Vector{Int}, iset[start:finish]) # Hack
    end

    # Creating the CSC sparse matrix.
    ptrs = convert(Vector{Int}, x.ptrs)
    for i in 1:length(ptrs)
        ptrs[i] = ptrs[i] + 1
    end
    rows = vcat(indices...)
    vals = vcat(values...)

    return SparseArrays.SparseMatrixCSC{Tv,Int}(x.dims[1], x.dims[2], ptrs, rows, vals)
end

