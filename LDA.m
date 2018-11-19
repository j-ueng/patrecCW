%for Question 3
function [Sb, Sw, rankSb, rankSw, W_lda] = LDA(tr_data, tr_label, W_pca, avg)
%Scatter matrices
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

rankSw = rank(Sw);
rankSb = rank(Sb);



end
