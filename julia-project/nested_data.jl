# nested_data.jl
# David Merrell
# 2020-02
#
# We frequently deal with data contained in nested
# Vectors and Dictionaries. This file contains useful
# functions for operating on this kind of data.


function get_at(data, key_vec::Vector)
    for k in key_vec
        data = data[k]
    end
    return data
end


function get_all_at(dict_vec::Vector, key_vec::Vector)
    results = []
    for d in dict_vec
        push!(results, get_at(d, key_vec))
    end
    return results
end


function set_at!(data, item, key_vec)
    if length(key_vec) == 1
        setindex!(data_structure, item, key_vec[1])
    else
        set_at!(data_structure[key_vec[1]], item, key_vec[2:end])
    end 
end



import Base: iterate
mutable struct LeafPathIterator
    data::Union{AbstractVector,AbstractDict}
    leaf_test::Function
end


function get_successors(path, data)
    if typeof(data) <: AbstractVector
        return [[path; [newk]] for newk in 1:length(data)]
    elseif typeof(data) <: AbstractDict
        return [[path; [newk]] for newk in keys(data)]
    else
        return []
    end
end


function iterate(lpi::LeafPathIterator, stack)
    while length(stack) > 0
        node_pth = pop!(stack)
        node = lpi.data
        for k in node_pth
            node = node[k]
        end
        if lpi.leaf_test(node)
            return node_pth, stack
        end
        succ = get_successors(node_pth, node)
        for s in succ
            push!(stack, s)
        end
    end
end

iterate(lpi::LeafPathIterator) = iterate(lpi, [[]])


function leaf_map(f::Function, data, leaf_test::Function)
    if leaf_test(data)
        return f(data)
    elseif typeof(data) <: AbstractDict
        return Dict(first(p) => leaf_map(f, last(p), leaf_test) for p in pairs(data))
    elseif typeof(data) <: AbstractVector
        return [leaf_map(f, x, leaf_test) for x in data]
    else
        return data
    end
end


function leaf_map!(f::Function, data, leaf_test::Function)
    for pth in LeafPathIterator(data, leaf_test)
        set_at!(data, f(get_at(data, pth)), pth)
    end
end


function aggregate_trees(agg::Function, data_vec::Vector, leaf_test::Function)
    if leaf_test(data_vec[1])
        return agg(data_vec)
    elseif typeof(data_vec[1]) <: AbstractDict
        return Dict(k => aggregate_trees(agg, [d[k] for d in data_vec], leaf_test) for k in keys(data_vec[1]) )
    elseif typeof(data_vec[1]) <: AbstractVector
        return [aggregate_trees(agg, [d[i] for d in data_vec], leaf_test) for i in keys(data_vec[1])]
    end
end



