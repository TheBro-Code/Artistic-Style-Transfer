%% Style_Transfer Function

function output = style_transfer(content_img, ...
                                 style_img, ...
                                 L_max, ...
                                 seg_mask, ...
                                 patch_sizes, ...
                                 sub_sampling_gap, ...
                                 IRLS_itr,I_alg,r)
                             
    %% Initialisation                          
    content_init = imhistmatch(content_img,style_img);
    
    %% Building Gaussian Pyramid of depth L_max
    
    content_pyramid = cell(L_max);
    content_pyramid{L_max} = content_init;
    
    style_pyramid = cell(L_max);
    style_pyramid{L_max} = style_img;
    
    seg_mask_pyramid = cell(L_max);
    seg_mask_pyramid{L_max} = seg_mask;
    
    for i = L_max - 1 : -1 : 1
    content_pyramid{i} = impyramid(content_pyramid{i+1},'reduce');
    style_pyramid{i} = impyramid(style_pyramid{i+1},'reduce');
    seg_mask_pyramid{i} = impyramid(seg_mask_pyramid{i+1},'reduce');
    end
    
    %% Building patch_matrices for style image for all L,n
    
    style_patch = cell(L_max,size(patch_sizes));
    
    for i = 1 : L_max
        for j = 1 : size(patch_sizes)
            img1 = im2col(style_pyramid{i}(:,:,1), ...
                    [patch_sizes(j),patch_sizes(j)]);
            img(:,:,1) = img1(:,:);
            
            img2 = im2col(style_pyramid{i}(:,:,2), ... 
                    [patch_sizes(j),patch_sizes(j)]);
            img(:,:,2) = img2(:,:);
           
            img3 = im2col(style_pyramid{i}(:,:,3), ... 
                    [patch_sizes(j),patch_sizes(j)]);
            img(:,:,3) = img3(:,:);
            
            style_patch{i,j} = img; 
        end
    end
    
    %% Initialise X as content_init + high noise
    
    X = content_pyramid{1} + sqrt(50)*randn(size(content_pyramid{1,1}));
    
    %%
    % Loop over scales 
    for i = 1:L_max
        % Loop over Patch-sizes
        for j = 1:size(patch_sizes)
            
            for k= 1:I_alg
                
                R_1 = im2col(X(:,:,1),[patch_sizes(j),patch_sizes(j)]);
                R_2 = im2col(X(:,:,2),[patch_sizes(j),patch_sizes(j)]);
                R_3 = im2col(X(:,:,3),[patch_sizes(j),patch_sizes(j)]);
                
                S = style_patch{i,j};
                
                R_new = [R_1,R_2,R_3];
                S_new = [S(:,:,1),S(:,:,2),S(:,:,3)];
                
                [Id,D] = knnsearch(S_new,R_new);
                
                
            end
            
        end
    end
    
    %%                         
end