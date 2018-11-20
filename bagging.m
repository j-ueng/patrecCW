function [classifiers, averages, dataBags, dataBagLabels] = bagging(tr_data, tr_label, nBags, nElements, M_pca, M_lda)
%1. Generate bootstrap replicates

%nBags: number of bags to generate
%nElements: number of elements (images) to put in each bag
%tr_data: input training data to split
%dataBag: sub-sample data used for training for every loop
%dataBagLabels: array with class labels for each sampled image

classifiers = cell(nBags, 1);
averages = cell(nBags, 1);
dataBags = cell(nBags, 1);
dataBagLabels = cell(nBags, 1);

for i = 1:nBags
    bag = randi(size(tr_data, 2), nElements, 1); 
    %returns a nElements x 1 sized array of integers between 1:size(tr_data,2)
    %these integers are the indices to pull from the full training data set
    %randi samples with replacement
    
    dataBag = zeros(size(tr_data,1), nElements);
    dataBagLabel = zeros(1, nElements); %these two are cleared and reset each loop
    
    %fill matrices with sampled images/class labels to use for training 
    for j = 1:nElements
        dataBag(:,j) = tr_data(:,bag(j));
        dataBagLabel(j) = tr_label(1, bag(j));
    end
    dataBags{i} = dataBag;
    dataBagLabels{i} = dataBagLabel;
    
%2. Train PCA-LDA classifier and store W_lda in classifiers for each bag
    [W_pca, avg] = PCA(dataBag, dataBagLabel, M_pca);   
    [W_lda] = LDA(dataBag, dataBagLabel, M_pca, W_pca, avg, M_lda);
    classifiers{i} = W_lda;
    averages{i} = avg;
end

end
