export SparseHDF5Matrix, sparse, issparse
import HDF5
import SparseArrays

"""
This class implements a `AbstractArray` wrapper around a sparse matrix in a HDF5 file.
We expect the 10X Genomics format where the matrix contents are stored in a group containing:

- `shape`, an integer dataset of length 2 containing the number of rows and columns in the matrix.
- `data`, a numeric dataset containing all of the non-zero elements in compressed sparse column (CSC) order.
- `indices`, an integer dataset containing the 0-based row indices of all non-zero elements in CSC order.
- `indptr`, an integer dataset containing the 0-based pointers into `indices` for the start and end of each column.

As with the [`DenseHDF5Array`](@ref), we recommend using instances of this class as a placeholder for a real matrix, rather than computing directly on it.
Once values are explicitly needed, the entire array can be loaded into memory with [`extractsparse`](@ref).
"""
struct SparseHDF5Matrix{Tv,Ti} <: HDF5Array{Tv, 2}
    file::String 
    name::String
    dims::Tuple{Integer,Integer}
    ptrs::Vector{Integer}

    # Mutable cache. Technically we need to protect this from multiple threads,
    # but whatever, it's not like we can do parallel HDF5 access anyway.
    cache::Dict{Integer,SparseArrays.SparseVector{Tv,Ti}}
end

"""
    SparseHDF5Matrix(file, name)

Create a `HDF5Array` for a sparse matrix inside a HDF5 file `file` at group `name`.
The group is expected to follow the 10X Genomics format, see [`SparseHDF5Matrix`](@ref) for more details.

# Examples
```jldoctest
julia> using HDF5Arrays

julia> tmp = tempname();

julia> examplesparse(tmp, "stuff", (20, 10), 0.2)

julia> x = SparseHDF5Matrix(tmp, "stuff");

julia> size(x)
(20, 10)

julia> using SparseArrays

julia> SparseArrays.issparse(x)
true
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

    return SparseHDF5Matrix{dtype,itype}(file, name, (dims...,), ptrs, Dict{Integer,SparseArrays.SparseVector{dtype,itype}}())
end

"""
    size(x::SparseHDF5Matrix{Tv,Ti})

Get the size of a `DenseHDF5Array{T,N}` as a `N`-tuple.
"""
function Base.size(x::SparseHDF5Matrix{Tv,Ti}) where {Tv,Ti}
    return x.dims
end

"""
    issparse(x::SparseHDF5Matrix{Tv,Ti})

Returns true.
"""
function SparseArrays.issparse(x::SparseHDF5Matrix{Tv, Ti}) where {Tv, Ti}
    return true 
end

"""
    getindex(x::SparseHDF5Matrix{Tv,Ti}, i, j)

Get the value of `x[i,j]`.
It would be unwise to use this function for anything other than `show()` - 
we suggest using [`extractsparse`](@ref) to obtain larger blocks of data.

# Examples
```jldoctest
julia> using HDF5Arrays

julia> tmp = tempname();

julia> examplesparse(tmp, "stuff", (20, 10), 0.2)

julia> x = SparseHDF5Matrix(tmp, "stuff");

julia> typeof(getindex(x, 1, 1))
Float64
```
"""
function Base.getindex(x::SparseHDF5Matrix{Tv,Ti}, i::Integer, j::Integer) where {Tv,Ti}
    if !haskey(x.cache, j)
        handle = HDF5.h5open(x.file)
        iset = handle[x.name * "/indices"]
        dset = handle[x.name * "/data"]

        start = Int(x.ptrs[j]) + 1 # 1-based indexing
        finish = Int(x.ptrs[j + 1]) # technically needs a -1, but this is cancelled by the +1 above

        dvals = dset[start:finish]
        ivals = iset[start:finish]
        for i in 1:length(ivals)
            ivals[i] += 1 # 1-based indexing.
        end
        vec = SparseArrays.sparsevec(ivals, dvals, x.dims[1])
        x.cache[j] = vec
    end

    return x.cache[j][i]
end

# TODO: use blockdim to determine whether we can extract multiple columns at
# once to reduce the number of calls.
function general_sparse_extractor(x::SparseHDF5Matrix{Tv,Ti}, indices, store::Function) where {Tv,Ti}
    handle = HDF5.h5open(x.file)
    iset = handle[x.name * "/indices"]
    dset = handle[x.name * "/data"]

    # Indices should be sorted and unique on input.
    allowed_rows = Set(indices[1])

    for ci in 1:length(indices[2])
        c = indices[2][ci]
        start = Int(x.ptrs[c]) + 1  # 1-based indexing
        finish = Int(x.ptrs[c + 1]) # technically needs a -1, but this is cancelled by the +1 above

        colvalues = dset[start:finish]
        colindices = iset[start:finish]
        for i in 1:length(colvalues)
            r = colindices[i] + 1 # 1-based indexing
            if r in allowed_rows
                store(r, c, colvalues[i])
            end
        end
    end
end

"""
    extractsparse(x, i, j; blockdim = nothing)

Extract an in-memory sparse matrix from a `SparseHDF5Matrix` `x`.
The returned matrix contains the same values as `x[i, j]`.

`blockdim` is currently ignored and is only provided for consistency with the method for `DenseHDF5Array`s.

# Examples
```jldoctest
julia> using HDF5Arrays

julia> tmp = tempname();

julia> examplesparse(tmp, "stuff", (20, 10), 0.2)

julia> x = SparseHDF5Matrix(tmp, "stuff");

julia> y = extractsparse(x, 1:5, [2,3,6,7]);

julia> size(y)
(5, 4)
```
"""
function extractsparse(x::SparseHDF5Matrix{Tv,Ti}, i, j; blockdim = nothing) where {Tv,Ti}
    indices = to_indices(x, (i, j))
    NR = length(indices[1])
    NC = length(indices[2])

    collated = create_back_mapping(indices)
    my_indices = sorted_unique_indices(collated)

    # Use of Int is a hack, we can't store the actual index here.
    store = Vector{Dict{Int,Tv}}(undef, x.dims[2])
    for d in my_indices[2]
        store[d] = Dict{Int,Tv}()
    end

    FUN = function (r, c, val)
        store[c][r] = val
    end

    general_sparse_extractor(x, (my_indices...,), FUN)

    return create_csc_matrix(NR, NC, collated[1], collated[2], store)
end

"""
    extractdense(x, i, j; blockdim = nothing)

Extract an in-memory dense `Matrix` from a `SparseHDF5Matrix`. 
The returned matrix contains the same values as `x[i, j]`.

`blockdim` is currently ignored and is only provided for consistency with the method for `DenseHDF5Array`s.

# Examples
```jldoctest
julia> using HDF5Arrays

julia> tmp = tempname();

julia> examplesparse(tmp, "stuff", (20, 10), 0.2)

julia> x = SparseHDF5Matrix(tmp, "stuff");

julia> y = extractdense(x, 1:5, [2,3,6,7]);

julia> size(y)
(5, 4)
```
"""
function extractdense(x::SparseHDF5Matrix{Tv,Ti}, i, j; blockdim = nothing) where {Tv,Ti}
    indices = to_indices(x, (i, j))
    NR = length(indices[1])
    NC = length(indices[2])

    collated = create_back_mapping(indices)
    my_indices = sorted_unique_indices(collated)

    # Use of Int is a hack, we can't store the actual index here.
    output = Array{Tv}(undef, (NR, NC))
    fill!(output, 0)

    FUN = function (r, c, val)
        output[collated[1][r], collated[2][c]] .= val
    end

    general_sparse_extractor(x, (my_indices...,), FUN)

    return output
end
