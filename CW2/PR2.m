% load('cuhk03_new_protocol_config_labeled.mat');
% features = jsondecode(fileread('feature_data.json'));

%baseline approach using k-nn classification
%1. performs k-nn on each vector in query set, finds nn in gallery set
%2. delete images/vector of considered ID with the same camID
%3. calculate error score for ranklist (successful if correct ID is
%anywhere within the ranklist
%use Euclidean distance


k_max = 10;
success = zeros(1, k_max);

for k = 1:k_max
    for q = 1:1400
        query = features(query_idx(q),:);
        for n = 1:length(gallery_idx)
            gallery(n, :) = features(n, :);
        end

        ranklist = knnsearch(gallery, query, 'K', k, 'Distance', 'euclidean');
        query_label = labels(query_idx(q));
    
        for n = 1:length(ranklist)
            gallery_label = labels(ranklist(n));
            if gallery_label == query_label 
                success(k) = success(k) + 1;
                break
            end
        end
    end
end

scores = success/length(query_idx);
