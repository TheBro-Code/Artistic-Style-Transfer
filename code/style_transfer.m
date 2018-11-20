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
    seg_mask_pyramid{L_max} = reshape(seg_mask,[],1);
    
    for i = L_max - 1 : -1 : 1
    content_pyramid{i} = impyramid(content_pyramid{i+1},'reduce');
    style_pyramid{i} = impyramid(style_pyramid{i+1},'reduce');
    seg_mask_pyramid{i} = impyramid(seg_mask_pyramid{i+1},'reduce');
    end
    
    %% Building patch_matrices for style image for all L,n
    
    style_patch = cell(L_max,size(patch_sizes, 2));
    for i = 1 : L_max
        for j = 1 : size(patch_sizes)
            
            
            
            img1 = im2col(style_pyramid{i}(:,:,1), ...
                    [patch_sizes(j),patch_sizes(j)]);
            
            img = zeros(size(img1,1),size(img1,2),3);
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
    
    X = double(content_pyramid{1}) + sqrt(50)*randn(size(content_pyramid{1,1}));
    X = reshape(X,[],1); % make X (3Nc x 1)
    
    %%
    % Loop over scales 
    for i = 1:L_max
        % Loop over Patch-sizes
        for j = 1:size(patch_sizes)
            X_hat = X;
            for k = 1:I_alg

                X_tilda = irls(X_hat,style_patch{i,j},patch_sizes(j),r,...
                    IRLS_itr,size(content_pyramid{i}));
                
                mask = repmat(seg_mask_pyramid{i}, 3, 1);
                X_hat = (X_tilda + mask.*content_img)/(mask + eye(size(mask)));
                X_hat = imhistmatch(reshape(X_hat, ...
                    size(content_pyramid{i})), style_pyramid{i});
                [thr,sorh,keepapp] = ddencmp('den','wv',X_hat);
                X_hat = wdencmp('gbl',X_hat,'sym4',2,thr,sorh,keepapp);
                X_hat = reshape(X_hat, [], 1);
                
            end
            
        end
        X_hat = reshape(X_hat, size(content_pyramid{i}));
        [m,n] = size(X_hat);
        X_hat = impyramid(X_hat, 'expand');
        if rem(m,2)==0 && rem(n,2)==0
            X_hat = imresize(X_hat,[2*m 2*n]);
        elseif rem(m,2)==0 || rem(n,2)==0
            if rem(m,2)==0
                X_hat = imresize(X_hat,[2*m 2*n-1]);
            else
                X_hat = imresize(X_hat,[2*m-1 2*n]);
            end
        end
    end
    
    output = X_hat;
    %%                         
end