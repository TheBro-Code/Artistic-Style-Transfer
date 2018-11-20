function X_tilda = irls(X,style_patch,patch_size,r,IRLS_itr,size)

    h = size(1);
    w = size(2);
    d = size(3);
    num_patches = (h-patch_size+1)*(w-patch_size+1);
    
     S_new = [style_patch(:,:,1); style_patch(:,:,2); style_patch(:,:,3)];
%     S_new((i-1)*patch_size*patch_size+1:i*patch_size*patch_size,:) = style_patch(:,:,i);
%     end
   
    X_itr = X;
    
    for i = 1:IRLS_itr
        
        X_itr = reshape(X_itr, size);
        
        R_1 = im2col(X_itr(:,:,1),[patch_size,patch_size]);
        R_2 = im2col(X_itr(:,:,2),[patch_size,patch_size]);
        R_3 = im2col(X_itr(:,:,3),[patch_size,patch_size]);
        R_new = [R_1;R_2;R_3];
        
        [Id,D] = nearest_neighbour(S_new,R_new);
        w_itr = D.^(r-2) + (1e-9);
        c_itr = sum(w_itr);
        
        term1 = zeros(d*w*h,d*w*h);
        term2 = zeros(d*w*h,1);
        
        for j = 1:num_patches
            R_j = patch_transform(size,patch_size,j);
            term1 = term1 + w_itr(j)*(R_j'*R_j);
            term2 = term2 + w_itr(j)*(R_j'*S_new(:,Id(j)));
        end
        
        term1 = term1 / c_itr;
        term2 = term2 / c_itr;
        
        X_itr = term1\term2;
        
    end
    
    X_tilda = X_itr;
    
end