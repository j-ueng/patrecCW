q = 60;
k = 5;
query = features(query_idx(q),:);
m = 1;
query_label = labels(query_idx(q));
discarded = 0;
for n = 1:length(gallery_idx)
    gallery_label = labels(gallery_idx(n));
      
    %exclude image if same label + same camID
    if gallery_label == query_label 
         if camId(gallery_idx(n)) ~= camId(query_idx(q))
                gallery(m, :) = features(n, :);
                m = m + 1;
         elseif camId(gallery_idx(n)) == camId(query_idx(q))
             discarded = discarded + 1;
         end
    elseif gallery_label ~= query_label
         gallery(m, :) = features(n, :);
         m = m + 1;
    end
end
m = m - 1;
ranklist = knnsearch(gallery, query, 'K', k, 'Distance', 'euclidean');

success = 0;
for n = 1:length(ranklist)
    show_gallery_label(n) = labels(ranklist(n));
end

for n = 1:length(ranklist)
    gallery_label = labels(ranklist(n));
    if gallery_label == query_label 
        success = 1;
        break
    end
end


%display query image / images in ranklist
figure(1);
imtitle = char(filelist(query_idx(q)));
im_q = imread(imtitle);
im_q = imresize(im_q, [323, 155]);
subplot(1,k + 1,1), imshow(im_q); 

%display images in ranklist
for i = 1:k
    imtitle = char(filelist(ranklist(i)));
    im_g = imread(imtitle);
    im_g = imresize(im_g, [323, 155]);
    subplot(1, k + 1, i + 1), imshow(im_g);
end
