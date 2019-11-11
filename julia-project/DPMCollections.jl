# DPMCollections.jl
# 2019-09-26
# David Merrell
#
# Simple collection type implementations.


module DPMCollections

import Base: isempty, pop!, push!, in, copy

export AbstractCollection, Stack, Queue

"""
    AbstractCollection{T}

Defines a simple container interface with typical methods.
"""
abstract type AbstractCollection{T} end
push!(ds::AbstractCollection{T}, item::T) where T = push!(ds.vec, item)
pop!(ds::AbstractCollection{T}) where T = pop!(ds.vec)
isempty(ds::AbstractCollection{T}) where T = isempty(ds.vec)
in(item::T, ds::AbstractCollection{T}) where T = in(item, ds.vec)

"""
    Queue{T}(vec::Vector{T})

Vector-based Queue. Implements AbstractCollection interface.
"""
mutable struct Queue{T} <: AbstractCollection{T}
    vec::Vector{T}
end
Queue{T}() where T = Queue{T}(Vector{T}())
push!(q::Queue{T}, item::T) where T = pushfirst!(q.vec, item)
copy(q::Queue{T}) where T = Queue(copy(q.vec))


"""
    Stack{T}(vec::Vector{T})

Vector-based Stack. Implements AbstractCollection interface.
"""
mutable struct Stack{T} <: AbstractCollection{T}
    vec::Vector{T}
end
Stack{T}() where T = Stack{T}(Vector{T}())
copy(s::Stack{T}) where T = Stack(copy(q.vec))


end
