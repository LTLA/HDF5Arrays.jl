export HDF5Array, extractsparse, extractdense, sparse
import HDF5
import SparseArrays

"""
The `HDF5Array` is an abstract type that describes the concept of a HDF5-backed array abstraction.
In other words, data is stored inside a HDF5 file and is retrieved on demand rather than being loaded into memory.
`T` is the type of the data while `N` is the dimensionality.

Concrete subtypes include the [`DenseHDF5Array`](@ref) and the [`SparseHDF5Matrix`](@ref).
Subtypes are expected to implement the [`extractdense`](@ref) and [`extractsparse`](@ref) methods.
They may also override the `SparseArrays.issparse` method to indicate whether they contain sparse data.

We provide conversion functions to quickly create in-memory `Array`s or `SparseMatrixCSC` objects from any `HDF5Array`s.
In addition, subsetting operations will automatically views on the original array rather than immediately loading data from file.
This enables lazy evaluation for memory-efficient operations on a subset of the dataset.

Currently, all `HDF5Array`s are read-only; calling `setindex!` will fail.
"""
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

"""
    Array(x::HDF5Array{T,N}})

Convert a `HDF5Array` into an in-memory `Array` of the same data type and dimension.
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
```
"""
function Array{T,N}(x::HDF5Array{T,N}) where {T,N}
    colons = Vector{Any}(undef, N)
    fill!(colons, :)
    return extractdense(x, colons...)
end

"""
    sparse(x::HDF5Array{T,N})

Convert a 2-dimensional `HDF5Array` into an in-memory sparse matrix of the same data type and dimension,
assuming that `T` is some numeric or boolean type.
This is equivalent to using [`extractsparse`](@ref) while requesting the full extent of each dimension.
Note that this only makes sense if `x` contains a high proportion of zeros.

# Examples
```jldoctest
julia> using HDF5Arrays, SparseArrays

julia> tmp = tempname();

julia> examplesparse(tmp, "stuff", (20, 10), 0.2)

julia> x = SparseHDF5Matrix(tmp, "stuff");

julia> y = sparse(x);

julia> typeof(y)
SparseMatrixCSC{Float64, Int64}
```
"""
function SparseArrays.sparse(x::HDF5Array{T,2}) where {T<:Union{Number,Bool}}
    return extractsparse(x, :, :)
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
    extractdense(x::SubArray{T,N,P<:HDF5Array{T,N},I,L}, indices...; blockdim = nothing)

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
    extractsparse(x::SubArray{T,N,P<:HDF5Array{T,N},I,L}, i, j; blockdim = nothing)

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

"""
    Array(x::SubArray{T,N,P<:HDF5Array{T,N},I,L})

Convert a view of a `HDF5Array` into an in-memory `Array` of the same type and dimension.
This is equivalent to using [`extractdense`](@ref) while requesting the full extent of each dimension.

# Examples
```jldoctest
julia> using HDF5Arrays

julia> tmp = tempname();

julia> exampledense(tmp, "stuff", (20, 10))

julia> x = DenseHDF5Array(tmp, "stuff");

julia> y = Array(x[1:10,2:5]);

julia> size(y)
(10, 4)
```
"""
function Array{T,N}(x::SubArray{T,N,P,I,L}) where {T,N,P<:HDF5Array{T,N},I,L}
    colons = Vector{Any}(undef, N)
    fill!(colons, :)
    return extractdense(x, colons...)
end

"""
    sparse(x::SubArray{T,2,P<:HDF5Array{T,2},I,L})

Convert a view of a 2-dimensional `HDF5Array` into an in-memory sparse matrix of the same type and dimension.
This is equivalent to using [`extractsparse`](@ref) while requesting the full extent of each dimension.

# Examples
```jldoctest
julia> using HDF5Arrays, SparseArrays

julia> tmp = tempname();

julia> examplesparse(tmp, "stuff", (20, 10), 0.2)

julia> x = SparseHDF5Matrix(tmp, "stuff");

julia> y = sparse(x[1:10,2:5]);

julia> typeof(y)
SparseMatrixCSC{Float64, Int64}
```
"""
function SparseArrays.sparse(x::SubArray{T,2,P,I,L}) where {T,P<:HDF5Array{T,2},I,L}
    return extractsparse(x, :, :)
end
