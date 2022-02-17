export HDF5Array
import HDF5

abstract type HDF5Array{T, N} <: AbstractArray{T, N} end

function Base.getindex(x::HDF5Array{T,N}, I...) where {T,N}
    return view(x, I...)
end

function Base.setindex!(x::HDF5Array{T,N}, v, I::Vararg{Integer,N}) where {T,N}
    throw(ErrorException(typeof(x) * " instances are strictly read-only"))
end
