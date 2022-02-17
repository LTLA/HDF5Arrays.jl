export DenseHDF5Array 
import HDF5

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
julia> x = DenseHDF5Array("zeisel-brain/dense.h5", "matrix")
```
"""
function DenseHDF5Array(file::String, name::String; cachedim = nothing)
    handle = HDF5.h5open(file)
    dset = handle[name]
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
