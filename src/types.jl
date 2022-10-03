Stream{T} = (Vector{Trace} where Trace <: AbstractVector{T}) where T <: AbstractFloat

MaybeStream{T} = Union{Missing, Stream{T}} where T <: AbstractFloat

Offsets = Vector{Int}

MaybeOffsets = Union{Missing, Offsets}