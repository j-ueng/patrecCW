%Pattern Recognition Coursework 1
%eigenvectors with non zero eigenvalues, M eigenvectors
function [V_sorted, D_sorted] = PCA(tr_data, M, nR, te_data, nC)

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

%application of eigenvectors
%face image reconstruction while varying M
im_c = tr_data(:,nR);
im = reshape(im_c, [56,46]);
im = imrotate(im.',270); 

weights = [1,M];
sum = zeros(size(im_c, 1), 1);
sub = im_c - avg_t;
for i=1:M
    weights(i) = dot(sub, V_sorted(:,i));
    sum = sum + (weights(i)*V_sorted(:,i));
end

im_r1 = avg_t + sum;
im_r = reshape(im_r1, [56,46]);
im_r = imrotate(im_r.',270); 

%display original and reconstructed
figure(2);
subplot(1,2,1),image(im);title('Original Image')
subplot(1,2,2),image(im_r);title(['Reconstructed Image, M = ', num2str(M)])




end

