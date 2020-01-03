function hill_dbn_wrapper(timeseries_filename, ref_graph_filename, output_filename, max_indegree, reg_mode)
%hill_dbn_wrapper Wraps around the dynamic_network_inference code from Hill
%et al. 2012.
%   Detailed explanation goes here

    ts_mat = read_timeseries_file(timeseries_filename);
    ref_adj = read_graph_file(ref_graph_filename);
    lambdas = 0.5:0.5:10.0;
    
    tic;
    [edge_probs, sign_mat, chosen_lambda] = dynamic_network_inference(ts_mat,...
                                                       max_indegree, ref_adj,... 
                                                       lambdas, reg_mode, 1, 0);
    elapsed = toc;
    
    write_to_file(output_filename, edge_probs, sign_mat, chosen_lambda, elapsed)

end


function ts_data = read_timeseries_file(timeseries_filename)

    ts_table = readtable(timeseries_filename);
    ts_data = {};
    u_ts = unique(ts_table.timeseries);
    
    for i=1:length(u_ts)
        rows = (ts_table.timeseries == u_ts(i));
        ts_data{u_ts(i)} = transpose(table2array(ts_table(rows, 3:end)));
    end

end


function ref_adj = read_graph_file(graph_filename)

    ref_adj_table = readtable(graph_filename);
    ref_adj = table2array(ref_adj_table);

end


function write_to_file(output_filename, edge_probs, sign_mat, chosen_lambda, elapsed)

    m = containers.Map({'edges', 'signs', 'lambda', 'time', 'edge_conf_key'},...
                       {edge_probs, sign_mat, chosen_lambda, elapsed, 'edges'});
    
    js = jsonencode(m);
    fid = fopen(output_filename, 'w');
    fprintf(fid, "%s", js);
    fclose(fid);
end
