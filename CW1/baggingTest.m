function [classifiers, averages, dataBagLabels] = baggingTest(tr_data, tr_label, nBags, nElements, M_pca, M_lda, te_data, te_label, n)

[classifiers, averages, dataBags, dataBagLabels] = bagging(tr_data, tr_label, nBags, nElements, M_pca, M_lda);

im_t = te_data(:,n);
w_t = zeros(1,M_lda);
class = zeros(1, nBags);

for i=1:nBags
    W_lda = classifiers{i};
    avg = averages{i};
    for j=1:M_lda %weights for the subtracted mean vector for the test im against every fisherface eigenvector
        w_t(j) = dot((im_t - avg),W_lda(:,j));
    end

    bag = dataBags{i};
    label = dataBagLabels{i};
    euDist = zeros(1, nElements);
    for j=1:nElements % i = image index in training data (dataBag)
        w_n = zeros(1, M_lda);
        for k=1:M_lda %j = eigenvector index in sorted matrix/number of weights
            w_n(k) = dot(bag(:,j) - avg, W_lda(:,k));
        end %completed weights vector for one training image
        %create vector of dists between test image weights and each training image weights
       
        euDist(j) = pdist2(w_t, w_n); %distance between test weights and weights of each training image stored here
    end
    [~, minDistIdx] = min(euDist);
    class(i) = label(minDistIdx);
end
class_final = majorityVote(class)
if te_label(1,n) == class_final
    disp('success');
else
    disp('failure');
end

end
