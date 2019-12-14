# dbn_visualization.jl
# 2019-11-12
# David Merrell
#
#

using Plots
Plots.pyplot()

#function visualize_digraph(dg::PSDiGraph, v_labels::Vector{String})
#    v = sort(collect(vertices(dg)))
#    ys = length(v):-1:1
#    y_dict = Dict([vert => ys[i] for (i,vert) in enumerate(v)])
#    xs = [1.0, length(v)]
#
#    p = Plots.plot(xlim=(0.0, length(v)+1), ylim=(0.0, length(v)+1),
#    legend=false, framestyle=:none, aspect_ratio=1.0)
#    Plots.yticks!(p, ys, v_labels)
#
#    for e in edges(dg)
#        Plots.plot!(p, xs, [y_dict[first(e)]; y_dict[last(e)]], c=:black)
#    end
#    for j=1:length(v)
#        Plots.scatter!(p, xs, [y_dict[v[j]]; y_dict[v[j]]], markersize=60.0/length(v), markercolor=:gray)
#        Plots.annotate!(p, xs[1]-0.2, y_dict[v[j]], Plots.text(v_labels[j], :right, 10))
#        Plots.annotate!(p, xs[2]+0.2, y_dict[v[j]], Plots.text(v_labels[j], :left, 10))
#    end
#
#    return p
#end
#
#function visualize_digraph_weighted(dg::PSDiGraph, weights::Dict, v_labels::Vector{String})
#    v = sort(collect(vertices(dg)))
#    ys = length(v):-1:1
#    y_dict = Dict([vert => ys[i] for (i,vert) in enumerate(v)])
#    xs = [1.0, length(v)]
#
#    p = Plots.plot(xlim=(0.0, length(v)+1), ylim=(0.0, length(v)+1),
#    legend=false, framestyle=:none, aspect_ratio=1.0)
#    Plots.yticks!(p, ys, v_labels)
#
#    for (e, w) in weights
#        Plots.plot!(p, xs, [y_dict[e[1]]; y_dict[e[2]]],
#                    color=Plots.RGBA(0,0,0, w), lw=3.0)
#    end
#    for j=1:length(v)
#        Plots.scatter!(p, xs, [y_dict[v[j]]; y_dict[v[j]]], markersize=60.0/length(v), markercolor=:gray)
#        Plots.annotate!(p, xs[1]-0.2, y_dict[v[j]], Plots.text(v_labels[j], :right, 10))
#        Plots.annotate!(p, xs[2]+0.2, y_dict[v[j]], Plots.text(v_labels[j], :left, 10))
#    end
#
#    return p
#end
#
#function animate_dbn_sampling(dg::PSDiGraph, v_labels, model, model_args, observations, proposal, proposal_args,
#                              involution, n_samples, thinning, title, gif_file_name)
#
#    tr, _ = Gen.generate(model, model_args, observations)
#    anim = @animate for i=1:n_samples
#        for t=1:thinning
#            tr, _ = Gen.metropolis_hastings(tr, proposal, proposal_args, involution)
#        end
#        p = visualize_digraph(tr[:G], v_labels)
#        curtitle = replace(title, "__SAMPLES__"=>i)
#        Plots.title!(p, curtitle)
#        if i%thinning == 0
#            println(i, " steps completed")
#        end
#    end
#    gif(anim, gif_file_name)
#
#end
#
#function animate_dbn_sampling_density(dg::PSDiGraph, v_labels, model, model_args, observations, proposal, proposal_args, involution,
#                                      n_samples, thinning, median_weight, title, gif_file_name)
#
#    tr, _ = Gen.generate(model, model_args, observations)
#    count_dict = Dict()
#
#    anim = @animate for i=1:n_samples
#
#        for t=1:thinning
#            tr, _ = Gen.metropolis_hastings(tr, proposal, proposal_args, involution)
#        end
#
#        G = tr[:G]
#	for e in edges(G)
#            count_dict[e] = get(count_dict, e, 0) + 1
#        end
#
#        weight_dict = Dict([e => (1.0*count/i)^(-1.0/log2(median_weight)) for (e, count) in count_dict])
#        #println(weight_dict)
#        p = visualize_digraph_weighted(G, weight_dict, v_labels)
#        curtitle = replace(title, "__SAMPLES__"=>i)
#        Plots.title!(p, curtitle)
#    end
#
#    fig = gif(anim, gif_file_name)
#
#    return count_dict
#end
#
#function graph_diff(new_g::PSDiGraph{T}, old_g::PSDiGraph{T}) where T
#    new_e = edges(new_g)
#    old_e = edges(old_g)
#
#    new_exc = setdiff(new_e, old_e)
#    old_exc = setdiff(old_e, new_e)
#    if length(new_exc) > 0 && length(old_exc) == 0
#        return "add", first(new_exc)
#    elseif length(old_exc) > 0 && length(new_exc) == 0
#        return "remove", first(old_exc)
#    elseif length(old_exc) > 0 && length(new_exc) > 0
#        return "reverse", (first(old_exc), first(new_exc))
#    else
#        return "none", nothing
#    end
#end
#
#function animate_dbn_exploration(dg::PSDiGraph, v_labels, model, model_args, observations, proposal, proposal_args,
#                                 involution, n_iterations, title, gif_file_name)
#
#    v = sort(collect(vertices(dg)))
#    ys = length(v):-1:1
#    y_dict = Dict([vert => ys[i] for (i,vert) in enumerate(v)])
#    xs = [1.0, length(v)]
#
#    tr, _ = Gen.generate(model, model_args, observations)
#    rejections = 0
#    anim = @animate for i=1:n_iterations
#        old_G = copy(tr[:G])
#        tr, acc = Gen.metropolis_hastings(tr, proposal, proposal_args, involution)
#        if !acc
#            rejections += 1
#        end
#        new_G = tr[:G]
#        p = visualize_digraph(new_G, v_labels)
#        curtitle = replace(title, "__STEPS__"=>i)
#        curtitle = replace(curtitle, "__REJECTIONS__"=>rejections)
#        Plots.title!(p, curtitle)
#        diffstr, e = graph_diff(new_G, old_G)
#        if diffstr == "add"
#            plot!(p, xs, [y_dict[e[1]]; y_dict[e[2]]], color=:blue, lw=4.0)
#        elseif diffstr == "remove"
#            plot!(p, xs, [y_dict[e[1]]; y_dict[e[2]]], color=:red, lw=4.0)
#        elseif diffstr == "reverse"
#            plot!(p, xs, [y_dict[e[1][1]]; y_dict[e[1][2]]], color=:yellow, lw=4.0)
#            plot!(p, xs, [y_dict[e[2][1]]; y_dict[e[2][2]]], color=:yellow, lw=4.0)
#        end
#
#    end
#    fig = gif(anim, gif_file_name)
#
#    return 1.0*rejections/n_iterations
#end


