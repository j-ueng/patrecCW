function [minDist_avg] = baggingTest(tr_data, tr_label, nBags, nElements, M_pca, M_lda, te_data, te_label, n)

[classifiers, averages] = bagging(tr_data, tr_label, nBags, nElements, M_pca, M_lda);

im_t = te_data(:,n);
size(im_t)
w_t = [1,M_lda];
minDistAll = zeros(nBags);

for i=1:nBags
    W_lda = classifiers{i};
    avg = averages{i};
    for j=1:M_lda
        w_t(j) = dot((im_t - avg),W_lda(:,j));
    end

    euDist = zeros(1, size(tr_data, 1));
    for j=1:size(tr_data, 2) % i = image index in training data
        w_n = [];
        for k=1:M %j = eigenvector index in sorted matrix/number of weights
            w_n(k) = dot(tr_data(:,j) - avg, W_lda(:,k));
        end %completed weights vector for one training image
        %create vector of dists between test image weights and each training image weights
        euDist(j) = pdist2(w_t, w_n);
    end
    minDist= min(euDist);
    minDistAll(i) = minDist;
end

minDist_avg = sum(minDistAll)/nBags;

end
