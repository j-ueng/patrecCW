%for Question 3
function [W_lda] = LDA(tr_data_uS, tr_label_uS, M_pca, W_pca, avg)
%Scatter matrices

%when using a randomly sampled subspace, must sort data/labels first
[tr_label0, ind1] = sort(tr_label_uS);
tr_label = []; 
for i=1:size(tr_label0, 2)
    if tr_label0(1, i) ~= 0
        tr_label(end + 1) = tr_label0(1, i);
    else
        break
    end
end

tr_data = [];
for i=1:size(ind1,2)
    tr_data(:,i) = tr_data_uS(:,ind1(i));
end

%creates vector of start indices of each class (from the training data set)
%indices in tr_label match with indices in tr_data
startIdx = [];
for i = 1:size(tr_label, 2)
    if i == 1
        startIdx(end + 1) = i;
    elseif tr_label(1,i) ~= tr_label(1,i-1)
        startIdx(end + 1) = i;
    end
end

classNumTotal = size(startIdx,2);
Sb = zeros(size(tr_data,1), size(tr_data,1));
num_class = zeros(1, classNumTotal);
mu_class = zeros(size(tr_data,1), classNumTotal);

%create the Between scatter matrix Sb
for i = 1:classNumTotal
    sum_c = zeros(size(tr_data,1),1);
    if i == classNumTotal
        num_im = 0;
        for j = startIdx(1,i):size(tr_data,2)
            num_im = num_im + 1;
            sum_c = sum_c + tr_data(:,j);
        end
    else
        num_im = 0;
        for j = startIdx(1,i):(startIdx(1,i+1)-1)
            num_im = num_im + 1;
            sum_c = sum_c + tr_data(:,j);
        end
    end
    num_class(i) = num_im;
    mu_class(:,i) = sum_c./num_im;
    Sb = Sb + (mu_class(:,i) - avg)*(mu_class(:,i) - avg).';
 
end

%create the Within scatter matrix Sw
Sw = zeros(size(tr_data,1), size(tr_data,1));
for i = 1:classNumTotal
    Sw_i = zeros(size(tr_data,1), size(tr_data,1));
    for j = 1:num_class(i)
        n = startIdx(i) - 1 + j;
        Sw_i = Sw_i + (tr_data(:,n) - mu_class(:,i))*(tr_data(:,n) - mu_class(:,i)).';
    end
    Sw = Sw + Sw_i;
end

%rank(Sw);
%rank(Sb);

%calculating W_lda (Fisherfaces)
W_pca = W_pca(:,1:M_pca);
[W_lda_unsorted, D0] = eig((W_pca.' * Sw * W_pca)*(W_pca.' * Sb * W_pca));

%ordering eigenvalues/eigenvectors in W_lda
D = zeros(size(D0,2), 1); %store eigenvalues in a col vector
for i=1:size(D0,2)
    for j=1:size(D0,1)
        if i==j
            D(i) = D0(i, j);
        end
    end
end

[D_sorted0, ind2] = sort(D, 'descend');
D_sorted = []; 
for i=1:size(D_sorted0, 1)
    if D_sorted0(i, 1) ~= 0
        D_sorted(end + 1, 1) = D_sorted0(i, 1);
    else
        break
    end
end

W_lda = [];
for i=1:size(ind2, 1)
    W_lda(:,i) = W_lda_unsorted(:,ind2(i));
end
%W_lda = W_lda(:,1:M_lda);

end
