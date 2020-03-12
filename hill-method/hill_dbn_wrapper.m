function hill_dbn_wrapper(timeseries_filename, ref_graph_filename, output_filename, max_indegree, reg_mode, timeout)
%hill_dbn_wrapper Wraps around the dynamic_network_inference code from Hill
%et al. 2012.
%   Detailed explanation goes here

    ts_mat = read_timeseries_file(timeseries_filename);
    ref_adj = read_graph_file(ref_graph_filename);
    V = size(ref_adj,1);
    lambdas = 1.0:0.5:15.0;
   
    % If max_indegree=-1, choose it automatically 
    % based on V. 
    if max_indegree == -1
        if V >= 1000
            max_indegree = 1;
        elseif V > 100
            max_indegree = 2;
        elseif V > 50
            max_indegree = 3;
        elseif V > 40
            max_indegree = 4;
        else
            max_indegree = 5;
        end
    end

    if reg_mode == "auto"
        if V >= 1000
            reg_mode = "linear";
        elseif V >= 100
            reg_mode = "quadratic";
        else
            reg_mode = "full";
        end
    end

    tic;
    [edge_probs, sign_mat, chosen_lambda, timed_out] = dynamic_network_inference(ts_mat,...
                                                       max_indegree, ref_adj,... 
                                                       lambdas, reg_mode, 1, 0, timeout);
    elapsed = toc;
    edge_probs = transpose(edge_probs);
    
    write_to_file(output_filename, edge_probs, sign_mat, chosen_lambda, elapsed, timed_out)

end


function ts_data = read_timeseries_file(timeseries_filename)

    ts_table = readtable(timeseries_filename);
    ts_table.timeseries = categorical(ts_table.timeseries);
    p = length(ts_table.Properties.VariableNames)-2;
    u_ts = unique(ts_table.timeseries);
    ts_data = {};

    for i=1:length(u_ts)
        rows = ts_table(ts_table.timeseries == u_ts(i),:);
        rows = sortrows(rows, 'timestep');
        rows = rows(:,3:end);
        t = height(rows);
        new_data = zeros(p,t);
        new_data(:,:) = transpose(table2array(rows));
        ts_data{i} = new_data;
    end
end


function ref_adj = read_graph_file(graph_filename)

    ref_adj_table = readtable(graph_filename);
    ref_adj = table2array(ref_adj_table);
    %ref_adj = (ref_adj > 0.25);

end


function write_to_file(output_filename, edge_probs, sign_mat, chosen_lambda, elapsed, timed_out)

    m = containers.Map({'edges', 'signs', 'lambda', 'time', 'edge_conf_key', 'timed_out'},...
                       {edge_probs, sign_mat, chosen_lambda, elapsed, 'edges', timed_out});
    
    js = jsonencode(m);
    fid = fopen(output_filename, 'w');
    fprintf(fid, "%s", js);
    fclose(fid);
end
