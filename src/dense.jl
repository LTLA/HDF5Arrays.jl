export DenseHDF5Array 
export sparse
import HDF5
import SparseArrays

"""
Wrapper class for a HDF5 array that has been loaded into memory.
"""
struct DenseHDF5Array{T, N} <: HDF5Array{T, N}
    file::String 
    name::String
    dims::Tuple{Vararg{Integer,N}}

    cachedim::Tuple{Vararg{Integer,N}}
    nblocks::Tuple{Vararg{Integer,N}}
    cache::Dict{Integer,Array{T,N}}
end

"""
    HDF5Array(file, name)

Create a `HDF5Array` for a dense dataset inside a HDF5 file `file` at name `name`.

# Examples
```jldoctest
julia> using HDF5Arrays

julia> x = DenseHDF5Array("zeisel-brain/dense.h5", "matrix")
```
"""
function DenseHDF5Array(file::String, name::String; cachedim = nothing)
    handle = HDF5.h5open(file)

    dset = handle[name]
    if !isa(dset, HDF5.Dataset)
        throw(ErrorException("'" * name * "' in '" * file * "' should be a dataset"))
    end

    dims = HDF5.get_extent_dims(dset)[1]
    type = HDF5.get_jl_type(dset)

    # Figuring out the caching dimensions. By default, we set it to 
    # the chunk dimensions (or 100x100x100,etc. if contiguous).
    if cachedim == nothing
        if HDF5.get_layout(HDF5.get_create_properties(dset)) == :chunked
            cachedim = HDF5.get_chunk(dset)
        else
            tmp = Vector{Int}(undef, length(dims))
            fill!(tmp, 100)
            cachedim = (tmp...,)
        end
    end

    N = length(dims)
    tmp = Vector{Int}(undef, N)
    for i in 1:N
        tmp[i] = Integer(ceil(dims[i]/cachedim[i]))
    end
    nblocks = (tmp...,)

    return DenseHDF5Array{type, N}(file, name, dims, cachedim, nblocks, Dict{Integer,Array{type,N}}())
end

function Base.size(x::DenseHDF5Array{T,N}) where {T,N}
    return x.dims
end

function Base.getindex(x::DenseHDF5Array{T,N}, I::Vararg{Integer,N}) where {T,N}
    block_id = 0
    multiplier = 1
    tmp = Vector{Int}(undef, N)
    for i in 1:N
        tmp[i] = Integer(floor((I[i] - 1) / x.cachedim[i]))
        block_id += multiplier * tmp[i]
        multiplier *= x.nblocks[i]
    end

    # Adding the block to the cache. This should probably be protected
    # by a mutex, but it's not like you can use HDF5 in parallel anyway!
    if !haskey(x.cache, block_id)
        handle = HDF5.h5open(x.file)
        dset = handle[x.name]

        indices = Vector{AbstractVector{Int}}(undef, N)
        for i in 1:N
            start = tmp[i] * x.cachedim[i] + 1
            finish = min(x.dims[i], start + x.cachedim[i] - 1)
            indices[i] = start:finish
        end

        x.cache[block_id] = Base.getindex(dset, indices...)
    end

    # Fetching the block and extracting the desired element.
    for i in 1:N
        tmp[i] = (I[i] - 1) % x.cachedim[i] + 1
    end
    return getindex(x.cache[block_id], tmp...)
end

"""
    Array(x)

Convert a `DenseHDF5Array` into an in-memory `Array` of the relevant type and dimension.
"""
function Array{T,N}(x::DenseHDF5Array{T,N}) where {T, N}
    handle = HDF5.h5open(x.file)
    dset = handle[x.name]
    return HDF5.read(dset)
end

"""
    sparse(x)

Convert a 2-dimensional `DenseHDF5Array` into an in-memory `SparseMatrix` of the relevant type,
assuming that the type is either numeric or boolean.
This is only really sensible when there is a high proportion of zero values.
"""
function sparse(x::DenseHDF5Array{T,2}) where {T<:Union{Number,Bool}}
    handle = HDF5.h5open(x.file)
    dset = handle[x.name]

    values = Vector{Vector{T}}(undef, x.dims[2])
    indices = Vector{Vector{Int}}(undef, x.dims[2])
    for i in 1:length(values)
        values[i] = Vector{T}()
        indices[i] = Vector{Int}()
    end

    # Extract by block, using the specified cache dimensions. This gets all
    # blocks down the columns, and then starts working across columns.
    for cb in 1:x.nblocks[2]
        cstart = (cb - 1) * x.cachedim[2] + 1
        cfinish = min(x.dims[2], cstart + x.cachedim[2] - 1)

        for rb in 1:x.nblocks[1]
            rstart = (rb - 1) * x.cachedim[1] + 1
            rfinish = min(x.dims[1], rstart + x.cachedim[1] - 1)

            # Adding everything to the cache.
            block = dset[rstart:rfinish,cstart:cfinish]
            for c in 1:size(block)[2]
                actual_c = c + cstart - 1
                curvals = values[actual_c]
                curinds = indices[actual_c]

                for r in 1:size(block)[1]
                    val = Base.getindex(block, r, c)
                    if val != 0
                        push!(curinds, r + rstart - 1)
                        push!(curvals, val)
                    end
                end
            end
        end
    end

    # Creating the CSC sparse matrix.
    ptrs = Vector{Int}(undef, x.dims[2] + 1)
    ptrs[1] = 1
    for i in 1:x.dims[2]
        ptrs[i + 1] = ptrs[i] + length(indices[i])
    end

    rows = vcat(indices...)
    vals = vcat(values...)
    return SparseArrays.SparseMatrixCSC{T,Int}(x.dims[1], x.dims[2], ptrs, rows, vals)
end
