function [faces, rError] = faceReconst(xbar, xn, u, M)

    im = reshape(xn, [56,46]);
    im = imrotate(im.',270); 

    faces = zeros(size(xn,1), 5);
    rError= [];
    
    for k = 1:size(M,2)
        w_r = [1,M(k)];
        sum = zeros(size(xn, 1), 1);
        sub = xn - xbar;
        for i=1:M(k)
            w_r(i) = dot(sub, u(:,i));
            sum = sum + (w_r(i)*u(:,i));
        end

        faces(:,k) = xbar + sum;
        rError(k) = norm(xn - faces(:,k));
    end

    %display original and reconstructed
    figure(4);
    subplot(2,3,1),image(im);title('Original Image')

    for i = 1:size(M,2)
        im_r = reshape(faces(:,i), [56,46]);
        im_r = imrotate(im_r.', 270);
        subplot(2,3,i+1),image(im_r);title(['M = ', num2str(M(i))])
    end
    
end
