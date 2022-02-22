export DenseHDF5Array, extractsparse, extractdense
import HDF5
import SparseArrays

"""
This class implements a `AbstractArray` wrapper around a HDF5 dataset.
A HDF5 dataset can thus be stored inside collections that expect an `AbstractArray`, without actually loading any data into memory.

It is also possible to perform calculations on this class, in which case values are retrieved from file on demand.
While memory-efficient, this approach is likely to be very slow, as it involves multiple (often redundant) I/O calls to the disk.

We recommend using this class as a placeholder for a real array in code that does not require the actual values.
Once values are explicitly needed, the entire array can be loaded into memory with [`extractdense`](@ref) or [`extractsparse`](@ref).
"""
struct DenseHDF5Array{T, N} <: HDF5Array{T, N}
    file::String 
    name::String
    dims::Tuple{Vararg{Integer,N}}

    # Caching utilities. This is only intended to avoid crushing performance
    # issues for getindex(), which is only to be used for giving something
    # sensible to show(). Actual computation should not use this cache.
    chunkdim::Tuple{Vararg{Integer,N}}
    nchunks::Tuple{Vararg{Integer,N}}
    cache::Dict{Integer,Array{T,N}}
end

function blocks_per_dim(blockdim::Tuple{Vararg{Integer}}, dims::Tuple{Vararg{Integer}})
    N = length(blockdim)
    tmp = Vector{Int}(undef, N)
    for i in 1:N
        tmp[i] = Integer(ceil(dims[i]/blockdim[i]))
    end
    return (tmp...,)
end

"""
    HDF5Array(file, name)

Create a `HDF5Array` for a dense dataset inside a HDF5 file `file` at name `name`.
No values are loaded into memory by the constructor (though `show`ing the matrix may load a few values for display).

# Examples
```jldoctest
julia> using HDF5Arrays

julia> tmp = tempname();

julia> exampledense(tmp, "stuff", (20, 10))

julia> x = DenseHDF5Array(tmp, "stuff");

julia> size(x)
(20, 10)
```
"""
function DenseHDF5Array(file::String, name::String)
    handle = HDF5.h5open(file)

    dset = handle[name]
    if !isa(dset, HDF5.Dataset)
        throw(ErrorException("'" * name * "' in '" * file * "' should be a dataset"))
    end

    dims = HDF5.get_extent_dims(dset)[1]
    type = HDF5.get_jl_type(dset)
    N = length(dims)

    # Figuring out the caching dimensions. By default, we set it to 
    # the chunk dimensions (or 100x100x100,etc. if contiguous).
    if HDF5.get_layout(HDF5.get_create_properties(dset)) == :chunked
        chunkdim = HDF5.get_chunk(dset)
    else
        tmp = Vector{Int}(undef, N)
        fill!(tmp, 100)
        chunkdim = (tmp...,)
    end

    nchunks = blocks_per_dim(chunkdim, dims)
    return DenseHDF5Array{type, N}(file, name, dims, chunkdim, nchunks, Dict{Integer,Array{type,N}}())
end

"""
    size(x::DenseHDF5Array{T,N})

Get the size of a `DenseHDF5Array{T,N}` as a `N`-tuple.
"""
function Base.size(x::DenseHDF5Array{T,N}) where {T,N}
    return x.dims
end

"""
    getindex(x::DenseHDF5Array{T,N}, I::Vararg{Integer,N})

Get the value of a `DenseHDF5Array{T,N}` at the position specified by indices `I`.
It would be unwise to use this function for anything other than `show()` - 
we suggest using [`extractdense`](@ref) to obtain larger blocks of data.

# Examples
```jldoctest
julia> using HDF5Arrays

julia> tmp = tempname();

julia> exampledense(tmp, "stuff", (20, 10))

julia> x = DenseHDF5Array(tmp, "stuff");

julia> typeof(getindex(x, 1, 1)) 
Float64
```
"""
function Base.getindex(x::DenseHDF5Array{T,N}, I::Vararg{Integer,N}) where {T,N}
    block_id = 0
    multiplier = 1
    tmp = Vector{Int}(undef, N)
    for i in 1:N
        tmp[i] = (I[i] - 1) ÷ x.chunkdim[i]
        block_id += multiplier * tmp[i]
        multiplier *= x.nchunks[i]
    end

    # Adding the block to the cache. This should probably be protected
    # by a mutex, but it's not like you can use HDF5 in parallel anyway!
    if !haskey(x.cache, block_id)
        handle = HDF5.h5open(x.file)
        dset = handle[x.name]

        indices = Vector{AbstractVector{Int}}(undef, N)
        for i in 1:N
            start = tmp[i] * x.chunkdim[i] + 1
            finish = min(x.dims[i], start + x.chunkdim[i] - 1)
            indices[i] = start:finish
        end

        x.cache[block_id] = Base.getindex(dset, indices...)
    end

    # Fetching the block and extracting the desired element.
    for i in 1:N
        tmp[i] = (I[i] - 1) % x.chunkdim[i] + 1
    end
    return getindex(x.cache[block_id], tmp...)
end

mutable struct BlockInfo
    range::AbstractRange{Int}
    internal::Vector{Int}
    function BlockInfo(j)
        new(1:0, Int[j])
    end
end

function general_dense_extractor(x::DenseHDF5Array{T,N}, indices, store::Function; blockdim = nothing) where {T,N}
    # Configuring the block dimensions.
    if blockdim == nothing
        blockdim = x.chunkdim
    end

    # Deciding which blocks have the indices we want. Note that indices should
    # already be sorted and unique prior to calling this function.
    idx = to_indices(x, indices)
    idx_by_block = Vector{Dict{Int,BlockInfo}}(undef, N)
    used_blocks = Vector{Vector{Int}}(undef, N)
    nblocks = Vector{Int}(undef, N)

    for d in 1:N
        curindex = idx[d]
        curblockdim = blockdim[d]
        curblocks = Dict{Int,BlockInfo}()

        for j in curindex
            b = (j - 1) ÷ curblockdim + 1
            if !haskey(curblocks, b)
                curblocks[b] = BlockInfo(j)
            else
                push!(curblocks[b].internal, j)
            end
        end

        # Indices should already be sorted and unique, so no need
        # to do any extra sorting on 'internal'.
        for (_, v) in curblocks
            first = v.internal[1]
            v.range = (first):(v.internal[length(v.internal)])
            for i in 1:length(v.internal)
                v.internal[i] -= first - 1
            end
        end

        idx_by_block[d] = curblocks
        used_blocks[d] = collect(keys(curblocks))
        sort!(used_blocks[d])
        nblocks[d] = length(curblocks)
    end

    # Extract along the fastest changing dimension. Normally in HDF5 this is the last,
    # but it seems Julia transposes things for us, so we'll go from the first.
    curblockpos = Vector{Int}(undef, N)
    for d in 1:N
        curblockpos[d] = nblocks[d]
    end

    curranges = Vector{AbstractRange{Int}}(undef, N)
    curidx = Vector{Vector{Int}}(undef, N)
    total_nblocks = prod(nblocks)

    handle = HDF5.h5open(x.file)
    dset = handle[x.name]

    for b in 1:total_nblocks
        # Incrementing the blocks.
        for d in 1:N
            if curblockpos[d] == nblocks[d]
                curblockpos[d] = 1
            else
                curblockpos[d] += 1
            end

            curblock = used_blocks[d][curblockpos[d]]
            curblockinfo = idx_by_block[d][curblock]
            curranges[d] = curblockinfo.range
            curidx[d] = curblockinfo.internal

            if curblockpos[d] != 1
                break
            end
        end

        block = getindex(dset, curranges...)

        # Extracting along the chosen block.
        curstarts = Vector{Int}(undef, N)
        curpos = Vector{Int}(undef, N)
        for d in 1:N
            curpos[d] = length(curidx[d])
            curstarts[d] = curranges[d][1] - 1
        end
        total = prod(curpos)
        innerpos = Vector{Int}(undef, N)
        realpos = Vector{Int}(undef, N)
        
        for v in 1:total
            for d in 1:N
                if curpos[d] == length(curidx[d])
                    curpos[d] = 1
                else
                    curpos[d] += 1
                end
                innerpos[d] = curidx[d][curpos[d]]
                realpos[d] = innerpos[d] + curstarts[d]
                if curpos[d] != 1
                    break
                end
            end

            val = getindex(block, innerpos...)
            store(realpos, val)
        end
    end
end

function create_back_mapping(indices)
    N = length(indices)
    collated = Vector{Dict{Int,Vector{Int}}}(undef, N)
    for d in 1:N
        current = Dict{Int,Vector{Int}}()
        curindices = indices[d]
        for i in 1:length(curindices)
            curdex = curindices[i]
            if haskey(current, curdex)
                push!(current[curdex], i)
            else
                current[curdex] = [i]
            end
        end
        collated[d] = current
    end
    return collated
end

function sorted_unique_indices(backmap)
    N = length(backmap)
    my_indices = Vector{AbstractVector{Int}}(undef, N) 
    for d in 1:N
        current = backmap[d]
        used = collect(keys(current))
        sort!(used)
        my_indices[d] = used
    end
    return my_indices
end

"""
    extractdense(x, I...; blockdim = nothing)

Extract an in-memory dense `Array` from a `DenseHDF5Array`. 
The returned array contains the same values as `x[I...]`.

For arbitrary indices, this function performs block-by-block extraction to reduce memory usage.
The size of each block is determined by `blockdim`, which should be a tuple of length equal to the number of dimensions.
If `blockdim` is not specified, the chunk dimensions are used instead; if the dataset is stored in a contiguous layout, arbitrary block dimensions are used.

Some optimization is applied if `I` only consists of `AbstractRange{Int}` values.

# Examples
```jldoctest
julia> using HDF5Arrays

julia> tmp = tempname();

julia> exampledense(tmp, "stuff", (20, 10))

julia> x = DenseHDF5Array(tmp, "stuff");

julia> y = extractdense(x, 1:5, 1:5);

julia> size(y)
(5, 5)

julia> y2 = extractdense(x, [1,3,5,7], [2,4,6,8,10]);

julia> size(y2)
(4, 5)
```
"""
function extractdense(x::DenseHDF5Array{T,N}, I...; blockdim = nothing) where {T,N}
    indices = to_indices(x, I)

    # Directly extracting if we have an abstract range.
    all_ars = true;
    for d in 1:N
        if !isa(indices[d], AbstractRange{Integer})
            all_ars = false
            break
        end
    end

    if all_ars
        handle = HDF5.h5open(x.file)
        dset = handle[x.name]
        return getindex(dset, indices...)
    end

    # Converting into a dictionary of back-mapped indices.
    collated = create_back_mapping(indices)
    my_indices = sorted_unique_indices(collated)

    # Creating a function to insert new values into the specified locations.
    lengths = [length(x) for x in indices]
    output = Array{T,N}(undef, (lengths...,))
    FUN = function (pos, val)
        setdex = Vector{AbstractVector{Int}}(undef, N)
        for d in 1:N
            setdex[d] = collated[d][pos[d]]
        end
        output[setdex...] .= val
    end

    # Performing the extraction.
    general_dense_extractor(x, (my_indices...,), FUN; blockdim = blockdim)
    return output
end

function create_csc_matrix(NR::Int, NC::Int, collated_rows::Dict{Int,Vector{Int}}, collated_cols::Dict{Int,Vector{Int}}, store::Vector{Dict{Int,T}}) where {T}
    rows = Vector{Vector{Int}}(undef, NC)
    vals = Vector{Vector{T}}(undef, NC)
    for (kc, vc) in collated_cols
        curcol = store[kc]

        tmp = Vector{Pair{Int,T}}()
        for (kr, vr) in curcol
            for r in collated_rows[kr]
                push!(tmp, Pair(r, vr))
            end
        end

        sort!(tmp)
        tmpr = [x.first for x in tmp]
        tmpv = [x.second for x in tmp]

        for c in vc
            rows[c] = tmpr
            vals[c] = tmpv
        end

        store[kc] = Dict{Int,T}() # let the GC clear the values.
    end

    ptrs = Vector{Int}(undef, NC + 1)
    ptrs[1] = 1
    for i in 1:NC
        ptrs[i + 1] = ptrs[i] + length(rows[i])
    end

    all_rows = vcat(rows...)
    rows = nothing # let the GC eat this up
    all_vals = vcat(vals...)
    vals = nothing

    return SparseArrays.SparseMatrixCSC{T,Int}(NR, NC, ptrs, all_rows, all_vals)
end

"""
    extractsparse(x, i, j; blockdim = nothing)

Extract an in-memory sparse matrix from a 2-dimensional `DenseHDF5Array`. 
The returned matrix contains the same values as `x[i, j]`.
This assumes that the type is either numeric or boolean, 
and is only useful when there is a high proportion of zero values in `x`.

For arbitrary indices, this function performs block-by-block extraction to reduce memory usage.
The size of each block is determined by `blockdim`, which should be a tuple of length equal to the number of dimensions.
If `blockdim` is not specified, the chunk dimensions are used instead; if the dataset is stored in a contiguous layout, arbitrary block dimensions are used.

# Examples
```jldoctest
julia> using HDF5Arrays

julia> tmp = tempname();

julia> exampledense(tmp, "stuff", (20, 10); density = 0.2)

julia> x = DenseHDF5Array(tmp, "stuff");

julia> y = extractsparse(x, 1:5, 1:5);

julia> typeof(y)
SparseArrays.SparseMatrixCSC{Float64, Int64}
```
"""
function extractsparse(x::DenseHDF5Array{T,2}, i, j; blockdim = nothing) where {T<:Union{Number,Bool}}
    indices = to_indices(x, (i, j))
    NR = length(indices[1])
    NC = length(indices[2])

    # Converting into a dictionary of back-mapped indices.
    collated = create_back_mapping(indices)
    my_indices = sorted_unique_indices(collated)

    # Creating a function to insert new values into the specified locations.
    store = Vector{Dict{Int, T}}(undef, x.dims[2])
    for d in my_indices[2]
        store[d] = Dict{Int,T}()
    end

    FUN = function (position, val)
        if val != 0
            store[position[2]][position[1]] = val
        end
    end

    general_dense_extractor(x, (my_indices...,), FUN; blockdim = blockdim)

    return create_csc_matrix(NR, NC, collated[1], collated[2], store)
end

"""
    Array(x)

Convert a `DenseHDF5Array` into an in-memory `Array` of the same type and dimension.
This is equivalent to using [`extractdense`](@ref) while requesting the full extent of each dimension.

# Examples
```jldoctest
julia> using HDF5Arrays

julia> tmp = tempname();

julia> exampledense(tmp, "stuff", (20, 10))

julia> x = DenseHDF5Array(tmp, "stuff");

julia> y = Array(x);

julia> size(y)
(20, 10)
"""
function Array{T,N}(x::DenseHDF5Array{T,N}) where {T, N}
    colons = Vector{Any}(undef, N)
    fill!(colons, :)
    return extractdense(x, colons...)
end

"""
    sparse(x)

Convert a 2-dimensional `DenseHDF5Array` into an in-memory `SparseMatrix` of the relevant type,
assuming that the type is either numeric or boolean.
This is only sensible when there is a high proportion of zero values in `x`.

# Examples
```jldoctest
julia> using HDF5Arrays

julia> tmp = tempname();

julia> exampledense(tmp, "stuff", (20, 10); density = 0.2)

julia> x = DenseHDF5Array(tmp, "stuff");

julia> y = sparse(x);

julia> typeof(y)
SparseArrays.SparseMatrixCSC{Float64, Int64}
```
"""
function sparse(x::DenseHDF5Array{T,2}) where {T<:Union{Number,Bool}}
    return extractsparse(x, :, :)
end
