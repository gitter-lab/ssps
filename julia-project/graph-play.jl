
include("DiGraph.jl")

using .DiGraphs

dag_edges = [["vehicle" "car"];
	     ["vehicle" "boat"];
	     ["vehicle" "airplane"];
	     ["vehicle" "train"];
	     ["airplane" "jet"];
	     ["airplane" "prop"];
	     ["airplane" "turboprop"];
	     ["turboprop" "jet-B"];
	     ["jet" "jet-A"];
	     ["jet" "jet-B"];
	     ["prop" "leaded"];
	     ["leaded" "gasoline"];
	     ["car" "gasoline"];
	     ["car" "electric"];
	     ["car" "diesel"];
	     ["boat" "diesel"];
	     ["boat" "uranium"];
	     ["train" "diesel"];
	     ["train" "coal"];
	     ["train" "electric"]]


dag = DiGraph(dag_edges)

println("\nDFS Traversal:\n",dfs_traversal(dag, "vehicle"))
println("\nBFS Traversal:\n",bfs_traversal(dag, "vehicle"))
println("\nis cyclic?\t", is_cyclic(dag))



