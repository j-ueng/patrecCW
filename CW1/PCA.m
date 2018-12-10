%Pattern Recognition Coursework 1
%eigenvectors with non zero eigenvalues, M eigenvectors
function [V_sorted, D_sorted, avg_t] = PCA(tr_data, tr_label, M, nR, te_data, te_label, nC)

    %mean image
    im_sum = zeros(size(tr_data,1),1);
    A = zeros(size(tr_data,1),size(tr_data,2));
    for i = 1:size(tr_data, 2)
        im_sum = im_sum + tr_data(:,i);
    end
    avg_t = im_sum./size(tr_data, 2);
    avg = reshape(avg_t, [56,46]);
    avg = imrotate(avg.',270); 
    figure(1);image(avg);title('Mean Image')

    tic;

    %eigenvalue & eigenvector matrices
    for i = 1:size(tr_data, 2)
       A(:,i) = tr_data(:,i) - avg_t;
    end
    S = 1/size(tr_data,2)*(A*A.');
    [V, D0] = eig(S); 
    D = zeros(size(D0,2), 1); %store eigenvalues in a col vector
    for i=1:size(D0,2)
        for j=1:size(D0,1)
            if i==j
                D(i) = D0(i, j);
            end
        end
    end
    
    %sorting, non zero eigenvalues
    [D_sorted0, ind] = sort(D, 'descend');
    D_sorted = []; 
    for i=1:size(D_sorted0, 1)
        if D_sorted0(i, 1) ~= 0
            D_sorted(end + 1, 1) = D_sorted0(i, 1);
        else
            break
        end
    end
    
    V_sorted = [];
    for i=1:size(ind, 1)
        V_sorted(:,i) = V(:,ind(i));
    end

    toc;

    %%
    % print first 20 eigenfaces
    figure(2);
    for i = 1:20
        eigface = reshape(V_sorted(:,i), [56,46]);
        eigface = imrotate(eigface.', 270);
        subplot(4,5,i),imshow(eigface,[]);
    end

    figure(3);
    plot([1:200],D_sorted(1:200));
    title('Trend of 200 largest eigenvalues');
    xlabel('Number of eigenvalues');
    ylabel('Eigenvalue');
    
end
