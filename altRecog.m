function [acc_rate,fail_idx] = altRecog(testd, testl, traind, trainl, M)

    % face recognition: reconstruction error

    succ_num = 0;
    succ_idx = [];
    fail_num = 0;
    fail_idx = [];

    for i = 1:size(testd,2)
        img_q = testd(:,i);

        tempdata = zeros(size(traind,1),8); % 8 datapoints per class
        euDist = [1, max(trainl)];
        for k = 1:max(trainl)
            % learn principal subspace for class i
            tempdata = traind(:,(8*k-7):(8*k));
            [V, D, xbar] = PCA2(tempdata,M);

            % reconstruct query image and compute reconstruction error
            face = faceRecon2(img_q, xbar, V, M);
            euDist(k) = norm(face - img_q);
        end

        [~, recogIdx] = min(euDist); % recogIdx = identity of the closest image

        % record success/failure instances for examples and acc rate
        if recogIdx == testl(i)
            succ_num = succ_num + 1;
            succ_idx(succ_num) = i;
        else
            fail_num = fail_num + 1;
            fail_idx(:,fail_num) = [testl(i); recogIdx];
        end
    end

    % compute accuracy rate
    acc_rate = succ_num / (succ_num + fail_num);

end
