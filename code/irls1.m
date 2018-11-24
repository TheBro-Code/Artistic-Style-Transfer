function X_tilda = irls1(X,style_patch,pca_out,patch_size,r,IRLS_itr,size_inp,sub_sampling_gap,R)


    h = size_inp(1);
    w = size_inp(2);
    d = size_inp(3);
    num_patches = (floor((h-patch_size)/sub_sampling_gap)+1)*(floor((w-patch_size)/sub_sampling_gap)+1);
        
    proj_mat = pca_out{1};
    S_eig = pca_out{2};
    mean_data = pca_out{3};
    
    X_itr = X;
    unsampled_pixs=double(~(sum(R,2)>0));
    
    for i = 1:IRLS_itr
        
        X_itr = reshape(X_itr, size_inp);
            
        R_new = zeros(3*patch_size*patch_size,num_patches);

        for j = 1:num_patches
            R_new(:,j) = X_itr(logical(R(:,j)));
        end
          
        R_new = R_new - repmat(mean_data,[1,size(R_new,2)]);
        R_eig = (proj_mat')*R_new;
        
        [Id,D] = nearest_neighbour(double(S_eig),double(R_eig));
        
        w_itr = (D+1e-9).^(r-2);
        
        % prevent black bars
        term1 = unsampled_pixs;
        term2 = X_itr(:).*unsampled_pixs;
                
        tic;
        for j = 1:num_patches
            R_j = double(R(:,j));
            tic;
            term1 = term1 + w_itr(j)*R_j;
            R_j(logical(R_j)) = style_patch(:,Id(j));
            term2 = term2 + w_itr(j)*R_j;
        end
        toc;
        
        X_itr = term2./(term1 + 1e-7);
        
    end
    
    X_tilda = X_itr;
    
end
