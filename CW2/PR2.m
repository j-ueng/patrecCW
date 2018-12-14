%load('cuhk03_new_protocol_config_labeled.mat');
%features = jsondecode(fileread('feature_data.json'));

%improved knn classification using LMNN
%1. create and train with vaildation set to choose mu
%2. delete images/vector of considered ID with the same camID
%3. calculate error score for ranklist (successful if correct ID is
%anywhere within the ranklist
%use Euclidean distance

% create validation set
train_label = labels(train_idx);
[rown,~] = size(unique(train_label));
val_label = train_label(randperm(rown));
val_label = val_label(1:100); % choose identity
val_idx = [];
vq_idx = [];
for k = 1:100
    temp  = find(labels == val_label(k));
    [tempsize,~] = size(temp);
    val_idx = [val_idx; temp(2:tempsize)];
    vq_idx = [vq_idx; temp(1)];
end
val_label = labels(val_idx);
val_feat = features(val_idx, :);
%val_feat = val_feat.';

% validation to find optimal maximum iteration

val_succ = zeros(1, 6);
query_used = 100;
i = 1;
k = 3; % knn classifier parameter
for maxiter = [10 20 30 40 50]
    
    % train on validation set
    [L,Det]=lmnnCG(val_feat.',val_label,4,'maxiter',maxiter);

    for q = 1:query_used
        vquery = features(vq_idx(q),:);
        vq_label = labels(vq_idx(q));
        m = 1;
        discarded = 0;
        for n = 1:length(val_idx)
            
            %exclude image if same label + same camID
            if val_label(n) == vq_label 
                if camId(val_idx(n)) ~= camId(query_idx(q))
                    val_feat(m, :) = features(val_idx(n), :);
                    store_val(m) = val_idx(n);
                    m = m + 1;
                else
                    discarded = discarded + 1;
                end
            elseif val_label(n) ~= vq_label
                val_feat(m, :) = features(val_idx(n), :);
                store_val(m) = val_idx(n);
                m = m + 1;
            end
        end
        fprintf('q = %d, discarded = %d\n', q, discarded);
        ranklist = knnsearch((L*val_feat.').', (L*vquery.').', 'K', k, 'Distance', 'euclidean');
        
        for n = 1:length(ranklist)
            val_label2 = labels(store_val(ranklist(n)));
            if val_label2 == vq_label 
                val_succ(1,i) = val_succ(1,i) + 1;
                break
            end
        end
    end
    i = i + 1;
end
valscores = val_succ/query_used;