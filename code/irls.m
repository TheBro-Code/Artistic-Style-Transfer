function X_tilda = irls(X,style_patch,patch_size,r,IRLS_itr,size)

    h = size(1);
    w = size(2);
    d = size(3);
    num_patches = (h-patch_size+1)*(w-patch_size+1);

    proj_mat = style_patch{1};
    S_eig = style_patch{2};
    mean_data = style_patch{3};
   
    X_itr = X;
    
    for i = 1:IRLS_itr
        
        i
        X_itr = reshape(X_itr, size);
        
        R_1 = im2col(X_itr(:,:,1),[patch_size,patch_size]);
        R_2 = im2col(X_itr(:,:,2),[patch_size,patch_size]);
        R_3 = im2col(X_itr(:,:,3),[patch_size,patch_size]);
        R_new = [R_1;R_2;R_3];
        
        R_new = R_new - repmat(mean_data,[1,size(R_new,2)]);
        R_eig = (proj_mat')*R_new;
        
        [Id,D] = nearest_neighbour(S_eig,R_eig);
        w_itr = D.^(r-2);
        
        term1 = zeros(1,d*w*h);
        term2 = zeros(d*w*h,1);
        
        for j = 1:num_patches
            j
            tic;
            R_j = patch_transform(size,patch_size,j);
            toc;
            tic;
            term1 = term1 + sum(R_j,1).*w_itr(j);
            toc;
            tic;
            term2 = term2 + w_itr(j)*(R_j'*S_eig(:,Id(j)));
            toc;
        end
        
        X_itr = (term2)./(term1');
        
    end
    
    X_tilda = X_itr;
    
end