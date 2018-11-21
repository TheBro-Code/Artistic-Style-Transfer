function X_tilda = irls(X,style_patch,patch_size,r,IRLS_itr,size_inp,sub_sampling_gap)


    h = size(1);
    w = size(2);
    d = size(3);
    num_patches = (floor((h-patch_size)/sub_sampling_gap)+1)*(floor((w-patch_size)/sub_sampling_gap)+1);

    proj_mat = style_patch{1};
    S_eig = style_patch{2};
    mean_data = style_patch{3};
   
    X_itr = X;
    
    for i = 1:IRLS_itr
        
        i
        X_itr = reshape(X_itr, size_inp);
        
        R_1 = im2col(X_itr(:,:,1),[patch_size,patch_size]);
        R_2 = im2col(X_itr(:,:,2),[patch_size,patch_size]);
        R_3 = im2col(X_itr(:,:,3),[patch_size,patch_size]);

        R_new = [R_1;R_2;R_3];
        R_new = reshape(R_new,3*patch_size*patch_size,(h-patch_size+1),(w-patch_size+1));
        R_new = R_new(:,1:sub_sampling_gap:end,1:sub_sampling_gap:end);
        R_new = reshape(R_new,3*patch_size*patch_size,[]);
        
        R_new = R_new - repmat(mean_data,[1,size(R_new,2)]);
        R_eig = (proj_mat')*R_new;
        
        [Id,D] = nearest_neighbour(S_eig,R_eig);
        w_itr = D.^(r-2);
        
        term1 = zeros(1,d*w*h);
        term2 = zeros(d*w*h,1);
                
        for j = 1:num_patches
            R_j = patch_transform(size_inp,patch_size,j,sub_sampling_gap);
            term1 = term1 + w_itr(j)*sum(R_j,1);
            term2 = term2 + w_itr(j)*(R_j'*S_eig(:,Id(j)));
        end
        X_itr = term2./(term1 + 1e-7)';
        
    end
    X_tilda = X_itr;
    
end