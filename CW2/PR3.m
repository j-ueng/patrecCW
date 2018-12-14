%load('cuhk03_new_protocol_config_labeled.mat');
%features = jsondecode(fileread('feature_data.json'));

% train_label = labels(train_idx);
% train_feat = features(train_idx, :);
% tic;
% [G, Det] = lmnnCG(train_feat.', train_label, 4, 'maxiter', 45);
% toc;

success = zeros(1, 3);
query_used = 1400;
i = 1;

for k = [1, 5, 10]
    for q = 1:query_used
        query = features(query_idx(q),:);
        query_label = labels(query_idx(q));
        m = 1;
        discarded = 0;
        for n = 1:length(gallery_idx)
            gallery_label = labels(gallery_idx(n));
            
            %exclude image if same label + same camID
            if gallery_label == query_label 
                if camId(gallery_idx(n)) ~= camId(query_idx(q))
                    gallery(m, :) = features(gallery_idx(n), :);
                    store_gallery(m) = gallery_idx(n);
                    m = m + 1;
                else
                    discarded = discarded + 1;
                end
            elseif gallery_label ~= query_label
                gallery(m, :) = features(gallery_idx(n), :);
                store_gallery(m) = gallery_idx(n);
                m = m + 1;
            end
        end
        fprintf('q = %d, discarded = %d\n', q, discarded);
        ranklist = knnsearch((G*gallery.').', (G*query.').', 'K', k, 'Distance', 'euclidean');
        
        for n = 1:length(ranklist)
            gallery_label = labels(store_gallery(ranklist(n)));
            if gallery_label == query_label 
                success(1,i) = success(1,i) + 1;
                break
            end
        end
    end
    i = i + 1;
end
scores = success/query_used;