# HDF5Arrays for Julia

## Overview 

This repository ports (parts of) Bioconductor's [**HDF5Array** package](https://bioconductor.org/packages/HDF5Array) to provide a HDF5-backed array abstraction in Julia.
Each `HDF5Array` instance only holds a pointer to the file in memory, and subsets of the data are retrieved on demand.
This enables users to manipulate large datasets with minimal memory consumption.
We support both dense arrays, represented as HDF5 datasets; and sparse matrices, represented using the [10X Genomics layout](https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/advanced/h5_matrices).

The `HDF5Array` class is implemented as a subtype of an `AbstractArray` and can (in theory) be used interchangeably with `Array`s for any read operations.
In practice, this abstraction is really only useful for mimicking the size and subsetting behavior of a real array.
If values are needed, it is often more efficient to manually extract large blocks of data from the file with `extractdense()` and `extractsparse()`,
rather than relying on `getindex()` to follow a sensible access pattern.

Write operations are not supported via the `HDF5Array` interface.

## Quick start

Users may install this package from the GitHub repository through the usual process on the Pkg REPL:

```julia
add https://github.com/LTLA/HDF5Arrays.jl
```

## Contact

This package is maintained by Aaron Lun ([**@LTLA**](https://github.com/LTLA)).
If you have bug reports or feature requests, please post them as issues at the [GitHub repository](https://github.com/LTLA/HDF5Arrays.jl/issues).
