# HDF5Arrays for Julia

## Overview 

This repository ports (parts of) Bioconductor's [**HDF5Array** package](https://bioconductor.org/packages/HDF5Array) to provide a HDF5-backed array abstraction in Julia.
Each `HDF5Array` instance only holds a pointer to the file in memory, and subsets of the data are retrieved on demand.
This enables users to manipulate large datasets with minimal memory consumption.
We support both dense arrays, represented as HDF5 datasets; and sparse matrices, represented using the [10X Genomics layout](https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/advanced/h5_matrices).

The `HDF5Array` class is implemented as a subtype of an `AbstractArray` and can (in theory) be used interchangeably with `Array`s for any read operations.
In practice, this abstraction is best used for mimicking the size and subsetting behavior of a real array.
If values are needed, it is usually more efficient to manually extract large blocks of data from the file with `extractdense()` and `extractsparse()`,
rather than relying on `Base.getindex()` to follow a sensible access pattern.

Write operations are not supported via the `HDF5Array` interface.

## Quick start

Users may install this package from the GitHub repository through the usual process on the Pkg REPL:

```julia
add https://github.com/LTLA/HDF5Arrays.jl
```

And then:

```julia
julia> using HDF5Arrays

julia> temp = tempname();

julia> exampledense(temp, "foo", (50, 100));

julia> x = DenseHDF5Array(temp, "foo");

julia> size(x)
(50, 100)

julia> y = Array(x);

julia> typeof(y)
Matrix{Float64} (alias for Array{Float64, 2})

julia> sub = x[1:10,5:20];

julia> size(sub)
(10, 16)
```

And the same for sparse matrices:

```julia>
julia> using HDF5Arrays, SparseArrays

julia> temp = tempname();

julia> examplesparse(temp, "foo", (50, 100), 0.1);

julia> x = SparseHDF5Matrix(temp, "foo");

julia> size(x)
(50, 100)

julia> y = SparseArrays.sparse(x);

julia> typeof(y)
SparseArrays.SparseMatrixCSC{Float64, Int64}

julia> sub = x[1:10,5:20];

julia> size(sub)
(10, 16)
```

## The `HDF5Array` class

### Class definition

```@docs
HDF5Array
```

### Subsetting

```@docs
getindex(x::HDF5Array{T,N}, I...) where {T,N}
```

### Extraction

```@docs
extractdense(x::SubArray{T,N,P,I,L}, indices...; blockdim = nothing) where {T,N,P<:HDF5Array{T,N},I,L}
```

```@docs
extractsparse(x::SubArray{T,2,P,I,L}, i, j; blockdim = nothing) where {T,P<:HDF5Array{T,2},I,L}
```

### Conversions

```@docs
Array{T,N}(x::HDF5Array{T,N}) where {T,N}
```

```@docs
SparseArrays.sparse(x::HDF5Array{T,2}) where {T<:Number}
```

```@docs
SparseArrays.sparse(x::SubArray{T,2,P,I,L}) where {T,P<:HDF5Array{T,2},I,L}
```

### Other methods

```@docs
setindex!(x::HDF5Array{T,N}, v, I::Vararg{Integer,N}) where {T,N}
```

## The `DenseHDF5Array` class

### Class definition

```@docs
DenseHDF5Array
```

### Constructor

```@docs
DenseHDF5Array(file::String, name::String)
```

### Basic methods

```@docs
size(x::DenseHDF5Array{T,N}) where {T,N}
```

```@docs
getindex(x::DenseHDF5Array{T,N}, I::Vararg{Integer,N}) where {T,N}
```

### Extraction

```@docs
extractdense(x::DenseHDF5Array{T,N}, I...; blockdim = nothing) where {T,N}
```

```@docs
extractsparse(x::DenseHDF5Array{T,2}, i, j; blockdim = nothing) where {T<:Union{Number,Bool}}
```

## The `SparseHDF5Matrix` class

### Class definition

```@docs
SparseHDF5Matrix
```

### Constructor

```@docs
SparseHDF5Matrix(file::String, name::String)
```

### Basic methods

```@docs
size(x::SparseHDF5Matrix{Tv,Ti}) where {Tv,Ti}
```

```@docs
getindex(x::SparseHDF5Matrix{Tv,Ti}, i::Integer, j::Integer) where {Tv,Ti}
```

```@docs
SparseArrays.issparse(x::SparseHDF5Matrix{Tv, Ti}) where {Tv, Ti}
```

### Extraction

```@docs
extractdense(x::SparseHDF5Matrix{Tv,Ti}, i, j; blockdim = nothing) where {Tv,Ti}
```

```@docs
extractsparse(x::SparseHDF5Matrix{Tv,Ti}, i, j; blockdim = nothing) where {Tv,Ti}
```

## Miscellaneous

```@docs
exampledense(x, name::String, dims::Tuple{Integer,Integer})
```

```@docs
examplesparse(x, name::String, dims::Tuple{Integer,Integer}, density::Float64)
```

## Contact

This package is maintained by Aaron Lun ([**@LTLA**](https://github.com/LTLA)).
If you have bug reports or feature requests, please post them as issues at the [GitHub repository](https://github.com/LTLA/HDF5Arrays.jl/issues).
