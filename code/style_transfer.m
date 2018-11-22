%% Style_Transfer Function

function output = style_transfer(content_img, ...
                                 style_img, ...
                                 hall_img, ...
                                 hall_coeff, ...
                                 L_max, ...
                                 seg_mask, ...
                                 patch_sizes, ...
                                 sub_sampling_gaps, ...
                                 IRLS_itr,I_alg,r)
    %% Initialisation
    disp('Initialising Variables');
    content_img = im2double(content_img);
    style_img = im2double(style_img);
    content_init = imhistmatch(content_img,style_img);
    scale_array = 2.^(L_max-1:-1:0);
    sigma_s = 5;
    sigma_r = 0.2;
    
    %% Building Gaussian Pyramid of depth L_max
    
    disp('Building Gaussian Pyramid');
    content_pyramid = cell(L_max,1);
    content_pyramid{L_max} = content_init;
    
    style_pyramid = cell(L_max,1);
    style_pyramid{L_max} = style_img;
    
    seg_mask_pyramid = cell(L_max,1);
    seg_mask_pyramid{L_max} = seg_mask;
    
    for i = L_max - 1 : -1 : 1
%     content_pyramid{i} = impyramid(content_pyramid{i+1},'reduce');
%     style_pyramid{i} = impyramid(style_pyramid{i+1},'reduce');
%     seg_mask_pyramid{i} = impyramid(seg_mask_pyramid{i+1},'reduce');
      content_pyramid{i} = imresize(content_init,1/scale_array(i));
      style_pyramid{i} = imresize(style_img,1/scale_array(i));
      seg_mask_pyramid{i} = imresize(seg_mask,1/scale_array(i));
    end
    
    %% Building patch_matrices for style image for all L,n
    disp('Building Patches for Style Matrix');
    style_patch = cell(L_max,length(patch_sizes));
    for i = 1 : L_max
        for j = 1 : length(patch_sizes)
            
            [h,w,~] = size(style_pyramid{i});
            
            img1 = im2col(style_pyramid{i}(:,:,1), ...
                    [patch_sizes(j),patch_sizes(j)]);
            
            img2 = im2col(style_pyramid{i}(:,:,2), ... 
                    [patch_sizes(j),patch_sizes(j)]);
           
            img3 = im2col(style_pyramid{i}(:,:,3), ... 
                    [patch_sizes(j),patch_sizes(j)]);
            
            img = [img1;img2;img3];
            img = reshape(img,3*patch_sizes(j)*patch_sizes(j),(h-patch_sizes(j)+1),(w-patch_sizes(j)+1));
            img = img(:,1:5:end,1:5:end);
            img = reshape(img,3*patch_sizes(j)*patch_sizes(j),[]);
%             pca_dimension = 75;
%             style_patch{i,j} = pca_reduction(img,pca_dimension);
            style_patch{i,j} = img;
             
        end
    end
    
    %% Initialise X as content_init + high noise
    disp('Initialising Content Image with High Noise');
    X = content_pyramid{1} + max(content_pyramid{1}(:))*randn(size(content_pyramid{1}));
    
    %%
    disp('Starting Style Transfer - ')
    % Loop over scales 
    X_hat = X;
    for i = 1:L_max
        
        hall_img_scaled = imresize(hall_img,1/scale_array(i));
        
        % Loop over Patch-sizes
        for j = 1:length(patch_sizes)
            
            if(i > 1 && j > 1)
                continue;
            end
            
            disp(strcat('Resolution Layer' , int2str(i) , ' , Patch Size' , int2str(patch_sizes(j))));
            for k = 1:I_alg
                    
                X_hat = reshape(X_hat,[],1); % make X (3Nc x 1)

                % Style Transfer
                X_hat = hall_coeff*hall_img_scaled(:) + (1 - hall_coeff)*X_hat;
                
                % Robust Aggregation
                X_tilda = irls(X_hat, ...
                               style_patch{i,j}, ...
                               patch_sizes(j),r,...
                               IRLS_itr, ...
                               size(content_pyramid{i}), ...
                               sub_sampling_gaps(j));
                
                % Content Fusion
%                 mask = repmat(reshape(seg_mask_pyramid{i},[],1), 3, 1);
%                 X_hat = (X_tilda + double(mask).*double(reshape(content_pyramid{i},[],1)))./(mask + ones(size(mask)));
                mask = repmat(1.5*seg_mask_pyramid{i}(:)/max(seg_mask_pyramid{i}(:)),3,1);
                X_hat = (X_tilda + double(mask).*double(reshape(content_pyramid{i},[],1)))./(mask + ones(size(mask)));
                
                % Color Transfer
                X_hat = imhistmatch(reshape(X_hat, ...
                    size(content_pyramid{i})), style_pyramid{i});
                
                % Denoising
%                 [thr,sorh,keepapp] = ddencmp('den','wv',X_hat);
%                 X_hat = wdencmp('gbl',X_hat,'sym4',2,thr,sorh,keepapp);
                X_hat = RF(X_hat, sigma_s, sigma_r);
                
            end
        end
        
        % Scale Up
        if (i~=L_max)
            X_hat = reshape(X_hat, size(content_pyramid{i}));
%             [m,n,~] = size(X_hat);
%             X_hat = impyramid(X_hat, 'expand');
%             if rem(m,2)==0 && rem(n,2)==0
%                 X_hat = imresize(X_hat,[2*m 2*n]);
%             elseif rem(m,2)==0 || rem(n,2)==0
%                 if rem(m,2)==0
%                     X_hat = imresize(X_hat,[2*m 2*n-1]);
%                 else
%                     X_hat = imresize(X_hat,[2*m-1 2*n]);
%                 end
%             end
            X_hat = imresize(X_hat,2);
        end
        
    end
    
    output = X_hat;    
end