# Containers.jl
# 2019-09-26
# David Merrell
#
# Simple container type implementations.


module Containers

import Base: isempty, pop!, push!, in, copy

export AbstractContainer, Stack, Queue

"""
    AbstractContainer{T}

Defines a simple container interface with typical methods.
"""
abstract type AbstractContainer{T} end
push!(ds::AbstractContainer{T}, item::T) where T = push!(ds.vec, item)
pop!(ds::AbstractContainer{T}) where T = pop!(ds.vec)
isempty(ds::AbstractContainer{T}) where T = isempty(ds.vec)
in(item::T, ds::AbstractContainer{T}) where T = in(item, ds.vec)

"""
    Queue{T}(vec::Vector{T})

Vector-based Queue. Implements AbstractContainer interface.
"""
mutable struct Queue{T} <: AbstractContainer{T}
    vec::Vector{T}
end
Queue{T}() where T = Queue{T}(Vector{T}())
push!(q::Queue{T}, item::T) where T = pushfirst!(q.vec, item)
copy(q::Queue{T}) where T = Queue(copy(q.vec))


"""
    Stack{T}(vec::Vector{T})

Vector-based Stack. Implements AbstractContainer interface.
"""
mutable struct Stack{T} <: AbstractContainer{T}
    vec::Vector{T}
end
Stack{T}() where T = Stack{T}(Vector{T}())
copy(s::Stack{T}) where T = Stack(copy(q.vec))


end
