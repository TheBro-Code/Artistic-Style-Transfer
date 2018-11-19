
function out = patch_transform(img_size,p)
[m, n] = img_size(:);
num_patches = (m - p + 1)*(n - p + 1);
R = zeros(p*p, m*n, num_patches);

    for r = 1: (m - p + 1)
        for c = 1: (n - p + 1)
            npatch = (c - 1) * (m - p + 1) + r;

            R_1 = zeros(p*p, m*n);
            for i = 1:p
                for j = 1:p
                    row = (j - 1) * p + i;
                    col = (c + j - 2) * m + r + i - 1;
                    R_1(row, col) = 1;
                end
            end

            R(:,:,npatch) = R_1(:,:);
        end
    end

out = R;

end