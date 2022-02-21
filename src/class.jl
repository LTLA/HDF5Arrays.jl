export HDF5Array,  extractsparse, extractdense
import HDF5

abstract type HDF5Array{T, N} <: AbstractArray{T, N} end

function Base.getindex(x::HDF5Array{T,N}, I...) where {T,N}
    return view(x, I...)
end

function Base.setindex!(x::HDF5Array{T,N}, v, I::Vararg{Integer,N}) where {T,N}
    throw(ErrorException(typeof(x) * " instances are strictly read-only"))
end

function sub_indices(x::SubArray{T,N,P,I,L}, indices...) where {T,N,P<:HDF5Array{T,N},I,L}
    sub = Vector{Any}(undef, N)
    idx = to_indices(x, indices)
    par = parentindices(x)
    for i in 1:N
        sub[i] = par[I[i]]
    end
    return sub
end

function extractdense(x::SubArray{T,N,P,I,L}, indices...; blockdim = nothing) where {T,N,P<:HDF5Array{T,N},I,L}
    sub = sub_indices(x, indices...)
    return extractdense(parent(x), sub...; blockdim = blockdim)
end

function extractsparse(x::SubArray{T,N,P,I,L}, indices...; blockdim = nothing) where {T,N,P<:HDF5Array{T,N},I,L}
    sub = sub_indices(x, indices...)
    return extractsparse(parent(x), sub...; blockdim = blockdim)
end
