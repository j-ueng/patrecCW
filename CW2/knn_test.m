q = 1400;
k = 5;
query = features(query_idx(q),:);
query_label = labels(query_idx(q));
m = 1;

for n = 1:length(gallery_idx)
    gallery_label = labels(gallery_idx(n));
            
    %exclude image if same label + same camID
        if gallery_label == query_label 
            if camId(gallery_idx(n)) ~= camId(query_idx(q))
                gallery(m, :) = features(gallery_idx(n), :);
                store_gallery(m) = gallery_idx(n);
                m = m + 1;
            end
        elseif gallery_label ~= query_label
            gallery(m, :) = features(gallery_idx(n), :);
            store_gallery(m) = gallery_idx(n);
            m = m + 1;
        end
end

ranklist = knnsearch(gallery, query, 'K', k, 'Distance', 'euclidean');
        
for n = 1:length(ranklist)
    gallery_label = labels(store_gallery(ranklist(n)));
        if gallery_label == query_label 
            success = 1;
            break
        end
end
for n = 1:length(ranklist)
    show_label(n) = labels(store_gallery(ranklist(n)));
end

%display query image / images in ranklist
figure(1);
imtitle = char(filelist(query_idx(q)));
im_q = imread(imtitle);
im_q = imresize(im_q, [323, 155]);
subplot(1,k + 1,1), imshow(im_q); 

%display images in ranklist
for i = 1:k
    imtitle = char(filelist(store_gallery(ranklist(i))));
    im_g = imread(imtitle);
    im_g = imresize(im_g, [323, 155]);
    subplot(1, k + 1, i + 1), imshow(im_g);
end
