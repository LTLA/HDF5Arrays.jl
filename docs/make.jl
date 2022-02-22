using Documenter
using HDF5Arrays
import SparseArrays

makedocs(
    sitename="HDF5Arrays.jl",
    modules = [HDF5Arrays],
    pages=[
        "Home" => "index.md"
    ]
)

deploydocs(;
    repo="github.com/LTLA/HDF5Arrays.jl",
)

