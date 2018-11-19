%Pattern Recognition Coursework 1
%PCA low dimensional computation
function [V_sorted, D_sorted] = PCA2(im_data, M, n)

    %mean image
    im_sum = zeros(size(im_data,1),1);
    A = zeros(size(im_data,1),size(im_data,2));
    for i = 1:size(im_data, 2)
        im_sum = im_sum + im_data(:,i);
    end
    avg_t = im_sum./size(im_data, 2);
    %avg = reshape(avg_t, [56,46]);
    %avg = imrotate(avg.',270);
    %figure(4);image(avg);title('Mean Image')

    tic;
    
    %eigenvalue & eigenvector matrices
    for i = 1:size(im_data, 2)
       A(:,i) = im_data(:,i) - avg_t;
    end
    S = 1/size(im_data,2)*(A.'*A);
    [V0, D0] = eig(S); 
    D = zeros(size(D0,2), 1); %store eigenvalues in a col vector
    for i=1:size(D0,2)
        for j=1:size(D0,1)
            if i==j
                D(i) = D0(i, j);
            end
        end
    end
    
    %sorting, non zero eigenvalues, and eigenvectors
    [D_sorted0, ind] = sort(D, 'descend');
    D_sorted = [];
    for i=1:size(D_sorted0, 1)
        if D_sorted0(i, 1) ~= 0
            D_sorted(end + 1, 1) = D_sorted0(i, 1);
        else
            break
        end
    end

    V = A * V0; % eig vec related by ui = A*vi
    V = V / norm(V);
    
    V_sorted = [];
    for i=1:size(ind, 1)
        V_sorted(:,i) = V(:,ind(i));
    end
    
    toc;
    
end
