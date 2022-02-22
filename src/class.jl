export HDF5Array, extractsparse, extractdense
import HDF5

abstract type HDF5Array{T, N} <: AbstractArray{T, N} end

"""
    getindex(x::HDF5Array{T,N}, I...)

Create a view into a `HDF5Array{T,N}` at the indices `I`.
This does not read any data from file, only acting as a delayed subsetting operation.
Calling [`extractdense`](@ref) or [`extractsparse`](@ref) on this view will use the indices to extract the desired subset.

# Examples
```jldoctest
julia> using HDF5Arrays

julia> tmp = tempname();

julia> exampledense(tmp, "stuff", (20, 10))

julia> x = DenseHDF5Array(tmp, "stuff");

julia> y = x[1:5, :];

julia> isa(y, SubArray)
true
```
"""
function Base.getindex(x::HDF5Array{T,N}, I...) where {T,N}
    return view(x, I...)
end

"""
    setindex!(x::HDF5Array{T,N}, v, I::Vararg{Integer,N}) 

All `HDF5Array`s are strictly read-only, so any attempt to call this function will throw an error.
"""
function Base.setindex!(x::HDF5Array{T,N}, v, I::Vararg{Integer,N}) where {T,N}
    throw(ErrorException(typeof(x) * " instances are strictly read-only"))
end

function sub_indices(x::SubArray{T,N,P,I,L}, indices...) where {T,N,P<:HDF5Array{T,N},I,L}
    sub = Vector{Any}(undef, N)
    idx = to_indices(x, indices)
    par = parentindices(x)
    for i in 1:N
        sub[i] = par[i][idx[i]]
    end
    return sub
end

"""
    extractdense(x, indices...; blockdim = nothing)

Extract an in-memory dense `Array` from a `SubArray` `x` of a parent `HDF5Array`.
The returned matrix contains the same values as `x[indices...]`, in addition to any subsetting used to create the view on the parent.

`blockdim` is forwarded to the method for the parent `HDF5Array`.

# Examples
```jldoctest
julia> using HDF5Arrays

julia> tmp = tempname();

julia> exampledense(tmp, "stuff", (20, 10))

julia> x = DenseHDF5Array(tmp, "stuff");

julia> y = x[2:10, [2,3,5,9,10]];

julia> z = extractdense(y, :, :);

julia> size(z)
(9, 5)
```
"""
function extractdense(x::SubArray{T,N,P,I,L}, indices...; blockdim = nothing) where {T,N,P<:HDF5Array{T,N},I,L}
    sub = sub_indices(x, indices...)
    return extractdense(parent(x), sub...; blockdim = blockdim)
end

"""
    extractsparse(x, i, j; blockdim = nothing)

Extract a sparse matrix from a `SubArray` `x` of a parent `HDF5Array`.
The returned matrix contains the same values as `x[i, j]`, in addition to any subsetting used to create the view on the parent.

`blockdim` is forwarded to the method for the parent `HDF5Array`.

# Examples
```jldoctest
julia> using HDF5Arrays

julia> tmp = tempname();

julia> examplesparse(tmp, "stuff", (20, 10), 0.1)

julia> x = SparseHDF5Matrix(tmp, "stuff");

julia> y = x[2:10, [2,3,5,9,10]];

julia> z = extractsparse(y, :, :);

julia> size(z)
(9, 5)
```
"""
function extractsparse(x::SubArray{T,N,P,I,L}, i, j; blockdim = nothing) where {T,N,P<:HDF5Array{T,N},I,L}
    sub = sub_indices(x, i, j)
    return extractsparse(parent(x), sub...; blockdim = blockdim)
end
