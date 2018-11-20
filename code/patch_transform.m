
function out = patch_transform(img_size,p,npatch)
m = img_size(1);
n = img_size(2);
d = img_size(3);

r = rem(npatch-1,m-p+1) + 1;
c = floor((npatch-1)/(m-p+1)) + 1;

R = zeros(p*p, m*n);
for i = 1:p
    for j = 1:p
        row = (j - 1) * p + i;
        col = (c + j - 2) * m + r + i - 1;
        R(row, col) = 1;
    end
end

    
out = zeros(d*p*p,d*m*n);

for i = 1:c
    out((i-1)*p*p + 1 :i*p*p,(i-1)*m*n + 1 : i*m*n) = R; 
end

end