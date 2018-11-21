
function out = patch_transform(img_size,p,npatch,sub_sampling_gap)
m = img_size(1);
n = img_size(2);
d = img_size(3);

r = rem(npatch-1,(floor((m-p)/sub_sampling_gap))+1)*sub_sampling_gap + 1;
c = floor((npatch-1)/(floor((m-p)/sub_sampling_gap)+1))*sub_sampling_gap + 1;

R = zeros(p*p,1);
for i = 1:p
    for j = 1:p
        row = (j - 1) * p + i;
        col = (c + j - 2) * m + r + i -1;
        R(row) = col;
    end
end

    
out = zeros(d*p*p,1);

for i = 1:d
    out((i-1)*p*p + 1 : i*p*p) = R + (i-1)*m*n; 
end

end