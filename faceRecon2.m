function face = faceRecon2(orig, xbar, u, M)

    w_r = [1,M];
    sum = zeros(size(orig, 1), 1);
    sub = orig - xbar;
    for i=1:M
        w_r(i) = dot(sub, u(:,i));
        sum = sum + (w_r(i)*u(:,i));
    end

    face = xbar + sum;

end
