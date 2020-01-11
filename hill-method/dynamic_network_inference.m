function [edge_prob_matrix interaction_sign_matrix chosen_lambda timed_out] = dynamic_network_inference(D,max_in_degree,prior_graph,lambdas,reg_mode,stdise,silent,timeout)
% [edge_prob_matrix interaction_sign_matrix chosen_lambda] = ...
%       dynamic_network_inference(D,max_in_degree,prior_graph,lambdas,reg_mode,stdise,silent)
%
% Implements exact dynamic Bayesian network inference with informative network priors, prior strength set
%       by empirical Bayes as described in Hill et al. (2012), to appear in Bioinformatics.
%
% Inputs:
% D: p x T x C data matrix or cell array of p x T x C data matrices. p = # variables (e.g. proteins), T = # time points, C = # time courses.
%    If cell array, network inference is performed on each data matrix separately with empirical Bayes performed on all data jointly.
% max_in_degree: Integer. Maximum in-degree (i.e. maximum number of parents a variable can have)
% prior_graph: binary p x p matrix. Adjacency matrix for network prior.
%              If empty ([]) or not input, a flat prior is used.
% lambdas: Vector of prior strength parameters to be considered in empirical Bayes analysis.
%          If a single value is input, this is taken as the prior strength (no empirical Bayes is performed).
%          Can be empty ([]) if using flat prior.
% reg_mode: String specifying regression model.
%           'full' - all interaction terms used (i.e up to product of all components in model)
%           'quadratic' - interaction terms up to pairwise products only
%           'linear' - no interaction terms
%           If left empty ([]) or not input, defaults to 'linear'.
% stdise: Binary value. If 1, standardises columns of design matrices and child (2nd time slice) variable to zero mean, unit variance.
%         Defaults to 1 if left empty ([]) or not input.
% silent: Binary value. If 0, progress is displayed on command window.
%         Defaults to 0 if left empty ([]) or not input.
% timeout: numeric value -- number of seconds of runtime permitted.
%
% Outputs:
% edge_prob_matrix: p x p matrix or cell array of p x p matrices. Entry (i,j) is exact posterior edge probability for edge
%                   e=(i,j) where i is variable in `previous' (first) time slice and j is variable in `current' (second) time slice
%                   Outputs cell array if network inference performed on multiple datasets (i.e. D is cell array).
% interaction_sign_matrix: p x p matrix or cell array of p x p matrices. Entry (i,j) is the sign of the correlation between variables i
%                          and j where i is variable in `previous' (first) time slice and j is variable in `current' (second) time slice
%                          Outputs cell array if network inference performed on multiple datasets (i.e. D is cell array).
% chosen_lambda: - Prior strength parameter selected by empirical Bayes
%
% version date: 31/7/12
% � Steven Hill and Sach Mukherjee, 2012

% Minor modifications made by David Merrell: 11/2019-01/2020

% set some default return values (in case of timeout)
edge_prob_matrix = cell(1,n_datasets);
interaction_sign_matrix = cell(1,n_datasets);
chosen_lambda = -1.0;
timed_out = false;

% start the clock
t_start = cputime;

% set default values and perform checks
if nargin<7 || isempty(silent)
    silent = 0;
end
if nargin<6 || isempty(stdise)
    stdise = 1;
end
if nargin<5 || isempty(reg_mode)
    reg_mode = 'linear';
end
if nargin<4 || isempty(lambdas)
    if nargin<3 || isempty(prior_graph)
        use_prior = 0;
        do_EB = 0;
        prior_graph = [];
        lambdas = [];
        if ~silent
            disp('Using a flat prior over graph space.')
        end
    else
        error('Prior graph has been input: Also need to input a prior strength value or a range of values for empirical Bayes.')
    end
elseif isempty(prior_graph)
    use_prior = 0;
    do_EB = 0;
else
    use_prior = 1;
    if ~silent
        disp('Using a network prior.')
    end
    if length(lambdas)==1
        do_EB = 0;
    else
        do_EB = 1;
    end
end

if strcmp(reg_mode,'linear')
    if ~silent
        disp('No interaction terms in regression model.')
    end
elseif strcmp(reg_mode,'quadratic')
    if ~silent
        disp('Pairwise interaction terms in regression model.')
    end
elseif strcmp(reg_mode,'full')
    if ~silent
        disp('All possible interaction terms (up to product of all parents) in regression model.')
    end
else
    error('Regression mode should be ''linear'',''quadratic'' or ''full''.')
end

if ~iscell(D)
    D = {D};
end
n_datasets = length(D);

if use_prior
    for i=2:length(D)
        if size(D{i},1)~=size(D{1},1)
            error('All datasets should consist of the same variables.')
        end
    end
    if size(D{1},1)~=size(prior_graph,1)
        error('Prior graph does not have correct number of variables.')
    end
end

% arrange data into correct form
for i=1:length(D)
    p = size(D{i},1);
    T = size(D{i},2);
    C = size(D{i},3);
    tmpD = zeros(2*p,(T-1)*C);
    for c=1:C
        tmpD(1:p,(c-1)*(T-1)+1:c*(T-1)) = D{i}(:,1:T-1,c);
        tmpD(p+1:2*p,(c-1)*(T-1)+1:c*(T-1)) = D{i}(:,2:T,c);
    end
    D{i} = tmpD;
end

if do_EB
    if ~silent
        if n_datasets>1
            disp('Performing empirical Bayes (EB) analysis over all datasets to select prior strength...')
        else
            disp('Performing empirical Bayes (EB) analysis to select prior strength...')
        end
    end
    
    EB_score = zeros(1,length(lambdas));
    EB_data = cell(n_datasets,1);
    for i=1:n_datasets
        if ~silent
            if n_datasets>1
                txt = strcat({'Dataset '},int2str(i),{'...'});
                disp(txt{1})
            end
        end
        EB_data{i}, tout = empirical_Bayes(D{i},max_in_degree,prior_graph,lambdas,reg_mode,stdise,silent, t_start, timeout);

        if tout
            timed_out = true;
            return
        end        

        EB_score = EB_score + EB_data{i}.EB_score;
    end
    idx = (EB_score==max(EB_score));
    chosen_lambda = lambdas(idx); chosen_lambda=chosen_lambda(1);
    edge_prob_matrix = cell(1,n_datasets);
    for i=1:n_datasets
        if ~silent
            if n_datasets>1
                txt = strcat({'Calculating posterior distribution over parent sets for dataset '},int2str(i),{'...'});
                disp(txt{1})
            else
                disp('Calculating posterior distribution over parent sets ...')
            end
        end
        scores, tout = posterior_scores(D{i},max_in_degree,prior_graph,chosen_lambda,reg_mode,stdise,silent,EB_data{i}.models,EB_data{i}.LL,EB_data{i}.Lpr_f, t_start, timeout);
        
        if tout
            timed_out = true;
            return
        end        

        if ~silent
            if n_datasets>1
                txt = strcat({'Calculating posterior edge probabilities for dataset '},int2str(i),{'...'});
                disp(txt{1})
            else
                disp('Calculating posterior edge probabilities...')
            end
        end
        edge_prob_matrix{i}, tout = edge_probabilities(scores,silent,t_start, timeout);
        if tout
            timed_out = true;
            return
        end        

    end
    
else
    chosen_lambda = lambdas;
    edge_prob_matrix = cell(1,n_datasets);
    for i=1:n_datasets
        if ~silent
            if n_datasets>1
                txt = strcat({'Calculating posterior distribution over parent sets for dataset '},int2str(i),{'...'});
                disp(txt{1})
            else
                disp(strcat('Calculating posterior distribution over parent sets...'))
            end
        end
        scores, tout = posterior_scores(D{i},max_in_degree,prior_graph,chosen_lambda,reg_mode,stdise,silent,[],[],[], t_start, timeout);
        if tout
            timed_out = true;
            return
        end        
        
        if ~silent
            if n_datasets>1
                txt = strcat({'Calculating posterior edge probabilities for dataset '},int2str(i),{'...'});
                disp(txt{1})
            else
                disp(strcat('Calculating posterior edge probabilities...'))
            end
        end
        edge_prob_matrix{i}, tout = edge_probabilities(scores,silent, t_start, timeout);
        
        if tout
            timed_out = true;
            return
        end        
    end
end 

interaction_sign_matrix = cell(1,n_datasets);
for i=1:n_datasets
    p = size(D{i},1);
    interaction_sign_matrix{i} = corrcoef(D{i}');
    interaction_sign_matrix{i} = sign(interaction_sign_matrix{i}(1:p/2,p/2+1:p));
end

if n_datasets==1
    edge_prob_matrix = edge_prob_matrix{1};
    interaction_sign_matrix = interaction_sign_matrix{1};
end

end

function EB_data, tout = empirical_Bayes(D,max_in_degree,prior_graph,lambdas,reg_mode,stdise,silent, t_start, timeout)

tout = false;

p = size(D,1)/2;
n = size(D,2);

% precalculate parts of the marginal log likelihood and standardised child ('response') vectors
LLpart1 = log(1+n);
LLpart2 = -n/2;
LLpart3 = n/(n+1);
if stdise
    if strcmp(reg_mode,'linear')
        D = standardise(D);
        Dstd = [];
    else
        D(p+1:2*p,:) = standardise(D(p+1:2*p,:));
        Dstd = standardise(D(1:p,:));
    end        
    LLpart4 = (n-1)*ones(1,p);
else
    LLpart4 = sum(D(p+1:2*p,:).^2,2);
    Dstd = zeros(p,0);
end

% number of possible parent sets
n_models = 1;
for i=1:max_in_degree
    n_models = n_models + nchoosek(p,i);
end

% calculate all possible parent sets
models = cell(1,n_models);
count = 0;
for d=0:max_in_degree
    nck = nchoosek(1:p,d);
    for i=1:size(nck,1);
        count = count+1;
        models{count} = nck(i,:);
    end
end

% calculate log marginal likelihood scores and network prior scores for all variables and parent sets
Lpr = zeros(n_models,p);
LL = zeros(n_models,p);
completeold = 0;
for i=1:n_models
    model = models{i};
    if stdise && strcmp(reg_mode,'linear')
        X = D(model,:)';
    else
        X = make_design_matrix(D(model,:),reg_mode,stdise,Dstd(model,:));
    end
    
    LLpart5 = [];
    LLpart6 = [];
    for s=1:p
        
        % if scoring empty model
        if isempty(X)
            X = zeros(n,0);
        end
        
        % log marginal likelihood
        [LL(i,s)  LLpart5 LLpart6] = log_marg_like(X,D(p+s,:)',LLpart1,LLpart2,LLpart3,LLpart4(s),LLpart5,LLpart6);
        
        % prior concordance function f = - (# parents in parent set but not in prior graph)
        idx = prior_graph(:,s)==0;
        Lpr(i,s) = -sum(idx(model));
    end
    
    % print progress
    if ~silent
        completenew=ceil(i/n_models*100);
        if rem(completenew,10)==0 && completeold~=completenew
            fprintf('%d %%..', completenew); completeold=completenew;
        end
    end

    % Check for timeout (DPM 2020)   
    if cputime - t_start > timeout
        tout = true;
        return
    end

end


% calculate prior scores and marginal likelihood empirical Bayes scores (P(data | lambda)) for various lambda
n_lambda = length(lambdas);
EB_score = zeros(1,n_lambda);
for l=1:n_lambda
    Lpr_tmp = lambdas(l)*Lpr;
    for s=1:p
        Lpr_tmp(:,s) = Lpr_tmp(:,s) - logsumexpv(Lpr_tmp(:,s));
    end
    
    % calculate EB_score
    EB_tmp = LL+Lpr_tmp;
    EB_tmp2 = zeros(p,1);
    for s=1:p
        EB_tmp2(s) = logsumexpv(EB_tmp(:,s));
    end
    EB_score(l) = sum(EB_tmp2);
end

idx = (EB_score==max(EB_score));
chosen_lambda = lambdas(idx);

EB_data.EB_score = EB_score;
EB_data.chosen_lambda = chosen_lambda;
EB_data.Lpr_f= Lpr; % prior concordance function scores
EB_data.LL = LL;    % log marginal likelihood scores
EB_data.models = models;  % parent sets
fprintf('\n')
end

function scores, tout = posterior_scores(D,max_in_degree,prior_graph,lambda,reg_mode,stdise,silent,models,LL,Lpr, t_start, timeout)
% calculates posterior scores over parent sets for each variable

tout = false;
p = size(D,1)/2;
n = size(D,2);

% precalculate parts of the marginal log likelihood and standardised data
LLpart1 = log(1+n);
LLpart2 = -n/2;
LLpart3 = n/(n+1);
if stdise
    if strcmp(reg_mode,'linear')
        D = standardise(D);
        Dstd = [];
    else
        D(p+1:2*p,:) = standardise(D(p+1:2*p,:));
        Dstd = standardise(D(1:p,:));
    end        
    LLpart4 = (n-1)*ones(1,p);
else
    LLpart4 = sum(D(p+1:2*p,:).^2,2);
    Dstd = zeros(p,0);
end

if isempty(models)
    
    % number of possible parent sets
    n_models = 1;
    for i=1:max_in_degree
        n_models = n_models + nchoosek(p,i);
    end
    
    % calculate all possible parent sets
    models = cell(1,n_models);
    count = 0;
    for d=0:max_in_degree
        nck = nchoosek(1:p,d);
        for i=1:size(nck,1);
            count = count+1;
            models{count} = nck(i,:);
        end
    end
    
    % calculate log marginal likelihood scores and network prior scores for all variables and parent sets
    Lpr = zeros(n_models,p);
    LL = zeros(n_models,p);
    completeold = 0;
    if ~isempty(prior_graph)
        for i=1:n_models
            model = models{i};
            if stdise && strcmp(reg_mode,'linear')
                X = D(model,:)';
            else
                X = make_design_matrix(D(model,:),reg_mode,stdise,Dstd(model,:));
            end
            
            LLpart5 = [];
            LLpart6 = [];
            for s=1:p
                
                % if scoring empty model
                if isempty(X)
                    X = zeros(n,0);
                end
                    
                % log marginal likelihood
                [LL(i,s)  LLpart5 LLpart6] = log_marg_like(X,D(p+s,:)',LLpart1,LLpart2,LLpart3,LLpart4(s),LLpart5,LLpart6);
                
                % prior concordance function f = - (# parents in parent set but not in prior graph)
                idx = prior_graph(:,s)==0;
                Lpr(i,s) = -sum(idx(model));
            end
            
            % print progress
            if ~silent
                completenew=ceil(i/n_models*100);
                if rem(completenew,10)==0 && completeold~=completenew
                    fprintf('%d %%..', completenew); completeold=completenew;
                end
            end
 
            % Check for timeout (DPM 2020)   
            if cputime - t_start > timeout
                tout = true;
                return
            end

        end
    else
        for i=1:n_models
            model = models{i};
            if stdise && strcmp(reg_mode,'linear')
                X = D(model,:)';
            else
                X = make_design_matrix(D(model,:),reg_mode,stdise,Dstd(model,:));
            end
            
            LLpart5 = [];
            LLpart6 = [];
            for s=1:p
                
                % if scoring empty model
                if isempty(X)
                    X = zeros(n,0);
                end
                    
                % log marginal likelihood
                [LL(i,s) LLpart5 LLpart6] = log_marg_like(X,D(p+s,:)',LLpart1,LLpart2,LLpart3,LLpart4(s),LLpart5,LLpart6);
            end
            
            % print progress
            if ~silent
                completenew=ceil(i/n_models*100);
                if rem(completenew,10)==0 && completeold~=completenew
                    fprintf('%d %%..', completenew); completeold=completenew;
                end
            end

            % Check for timeout (DPM 2020)   
            if cputime - t_start > timeout
                tout = true;
                return
            end

 
        end
    end
        
end

if ~isempty(prior_graph)
    Lpr = lambda*Lpr;
    LP = LL + Lpr;
    for s=1:p
        Lpr(:,s) = Lpr(:,s) - logsumexpv(Lpr(:,s));
    end
else
    LP = LL;
    Lpr = [];
end

for s=1:p
    LP(:,s) = LP(:,s) - logsumexpv(LP(:,s));
end
   
scores.models = models;
scores.LL = LL;
scores.Lpr = Lpr;
scores.LP = LP;
if ~silent
    fprintf('\n')
end
end

function X = make_design_matrix(X,reg_mode,stdise,Xstd)
% Calculates deign matrices and standardises if stdise==1 (i.e. Xstd is not empty)

[d,n] = size(X);

switch reg_mode
    case 'linear'
        X = [ones(1,n);X];
        
    case 'quadratic'
        if ~stdise
            count = d;
            for i=1:d
                for j=i+1:d
                    count = count+1;
                    X(count,:) = X(i,:).*X(j,:);
                end
            end
            X = [ones(1,n);X];
        else
            count = 0;
            if d>=2
                Xtmp = zeros(nchoosek(d,2),n);
            else
                Xtmp = [];
            end
            for i=1:d
                for j=i+1:d
                    count = count+1;
                    Xtmp(count,:) = X(i,:).*X(j,:);
                end
            end
            X = [Xstd;standardise(Xtmp)];
        end
        
    case 'full'
        if ~stdise
            count = d;
            for i=2:d
                idx = nchoosek(1:d,i);
                for j=1:size(idx,1)
                    count = count+1;
                    X(count,:) = prod(X(idx(j,:),:),1);
                end
            end
            X = [ones(1,n);X];
        else
            count = 0;
            Xtmp = zeros(2^d-d-1,n);
            for i=2:d
                idx = nchoosek(1:d,i);
                for j=1:size(idx,1)
                    count = count+1;
                    Xtmp(count,:) = prod(X(idx(j,:),:),1);
                end
            end
            X = [Xstd;standardise(Xtmp)];
        end
end
X = X';
end

function [LL A cnt]= log_marg_like(X,Y,LLpart1,LLpart2,LLpart3,LLpart4,A,cnt)

d = size(X,2);
max_iterations = 1000;

if isempty(A)
    
    % matrix inversion conditioning parameters
    lambda_r = 0.001;
    lambda_t = 10^-4;
    
    A = X'*X;
    cnd = rcond(A);
    
    if cnd>lambda_t
        LL = (-d/2)*LLpart1 + LLpart2*log(LLpart4 - LLpart3*Y'*X*(X\Y));
        cnt = 0;
    else
        cnt = 1;
        while cnt<max_iterations
            % matrix conditioning - Tikhonov regularization
            A = A+lambda_r.*eye(d);
            cnd = rcond(A);
            if cnd>lambda_t
                LL = (-d/2)*LLpart1 + LLpart2*log(LLpart4 - LLpart3*Y'*X*(A\X')*Y);
                break
            end
            cnt = cnt+1;
        end
        if cnt==max_iterations
            LL = (-d/2)*LLpart1 + LLpart2*log(LLpart4- LLpart3*Y'*X*pinv(A)*X'*Y);
        end
    end
else
    if cnt==0
        LL = (-d/2)*LLpart1 + LLpart2*log(LLpart4 - LLpart3*Y'*X*(X\Y));
    elseif cnt<max_iterations
        LL = (-d/2)*LLpart1 + LLpart2*log(LLpart4 - LLpart3*Y'*X*(A\X')*Y);
    else
        LL = (-d/2)*LLpart1 + LLpart2*log(LLpart4- LLpart3*Y'*X*pinv(A)*X'*Y);
    end
end
end

function edge_prob_matrix, tout = edge_probabilities(scores,silent, t_start, timeout)

tout = false;
p = size(scores.LL,2);

edge_prob_matrix = zeros(p);

completeold = 0;
for s=1:p
    tmp_lists = cell(1,p);
    for i=1:length(scores.models)
        m = scores.models{i};
        LP = scores.LP(i,s);
        for j=1:length(m)
            tmp_lists{m(j)}(end+1) = LP;
        end
    end
    for s1=1:p
        if ~isempty(tmp_lists{s1})
            edge_prob_matrix(s1,s) = exp(logsumexpv(tmp_lists{s1}));
        end
    end
    % print progress
    if ~silent
        completenew=ceil(s/p*100);
        if rem(completenew,10)==0 && completeold~=completenew
            fprintf('%d %%..', completenew); completeold=completenew;
        end
    end
    % Check for timeout (DPM 2020)   
    if cputime - t_start > timeout
        tout = true;
        return
    end
end
if ~silent
    fprintf('\n')
end
end

function D = standardise(D)
% Input:
% D - Matrix of size d x n. d = # of variables. n = number of samples.
%
% Output:
% D - Matrix of size d x n. Standardised version of D. Each row has zero mean and unit variance.

D = D-(repmat(mean(D,2),1,size(D,2)));
D = D./(repmat(std(D,0,2),1,size(D,2)));
end

function s = logsumexpv(a)
% Returns log(sum(exp(a))) while avoiding numerical underflow.
%
% e.g., log(e^a1 + e^a2) = a1 + log(1 + e^(a2-a1)) if a1>a2
% If a1 ~ a2, and a1>a2, then e^(a2-a1) is exp(small negative number),
% which can be computed without underflow.

a = a(:)'; % make row vector
m = max(a);
b = a - m*ones(1,length(a));
s = m + log(sum(exp(b)));
end
