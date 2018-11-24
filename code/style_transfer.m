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
    pca_dimension = 75;
    
    %% Building Gaussian Pyramid of depth L_max
    
    disp('Building Gaussian Pyramid');
    content_pyramid = cell(L_max,1);
    content_pyramid{L_max} = content_init;
    
    style_pyramid = cell(L_max,1);
    style_pyramid{L_max} = style_img;
    
    seg_mask_pyramid = cell(L_max,1);
    seg_mask_pyramid{L_max} = seg_mask;
    
    for i = L_max - 1 : -1 : 1
      content_pyramid{i} = imresize(content_init,1/scale_array(i));
      style_pyramid{i} = imresize(style_img,1/scale_array(i));
      seg_mask_pyramid{i} = imresize(seg_mask,1/scale_array(i));
    end
    
    %% Initialise X as content_init + high noise
    disp('Initialising Content Image with High Noise');
    X = content_pyramid{1} + 0.2*max(content_pyramid{1}(:))*randn(size(content_pyramid{1}));
    
    %%
    disp('Starting Style Transfer - ')
    % Loop over scales 
    X_hat = X;
    for i = 1:L_max
        
        hall_img_scaled = imresize(hall_img,1/scale_array(i));
        
        % Loop over Patch-sizes
        for j = 1:length(patch_sizes)
            
            if(i ~= L_max && j > 2)
                continue;
            end
            
            disp(strcat('Resolution Layer' , int2str(i) , ' , Patch Size' , int2str(patch_sizes(j))));
            
            [h,w,d] = size(content_pyramid{i});
            
            % Building Patch Transformation Matrix
            num_patches = (floor((h-patch_sizes(j))/sub_sampling_gaps(j))+1)*(floor((w-patch_sizes(j))/sub_sampling_gaps(j))+1);
            R = false(h*w*d,num_patches);
            for row = 1:sub_sampling_gaps(j):h-patch_sizes(j)+1
                for col = 1:sub_sampling_gaps(j):w-patch_sizes(j)+1
                    
                    x = (col-1)/sub_sampling_gaps(j);
                    y = floor((h-patch_sizes(j))/sub_sampling_gaps(j))+1;
                    z = (row-1)/sub_sampling_gaps(j) + 1;
                    patch_num = x*y + z;

                    sample_patch = false(h,w,d);
                    sample_patch(row:row+patch_sizes(j)-1,col:col+patch_sizes(j)-1,:) = true;
                    R(:,patch_num) = sample_patch(:);
                    
                end
            end
            
            % Building Style Patch 
            p_size = patch_sizes(j)*patch_sizes(j);
  
            style_patch = zeros(3*p_size, ...
                (h-patch_sizes(j)+1)*(w-patch_sizes(j)+1));
            
            style_patch(1:p_size, :) = im2col(style_pyramid{i}(:,:,1), ...
                    [patch_sizes(j),patch_sizes(j)]);
            
            style_patch(p_size+1:2*p_size,:) = im2col(style_pyramid{i}(:,:,2), ... 
                    [patch_sizes(j),patch_sizes(j)]);
            
            style_patch(2*p_size+1:3*p_size, :) = im2col(style_pyramid{i}(:,:,3), ... 
                    [patch_sizes(j),patch_sizes(j)]);
                
            style_patch = reshape(style_patch,3*patch_sizes(j)*patch_sizes(j),...
                                 (h-patch_sizes(j)+1),(w-patch_sizes(j)+1));
          
            style_patch = style_patch(:,1:5:end,1:5:end);
            style_patch = reshape(style_patch,3*patch_sizes(j)*patch_sizes(j),[]);
            
            pca_out = pca_reduction(style_patch,pca_dimension);
            
            for k = 1:I_alg
                
                X_hat = reshape(X_hat,[],1); % make X (3Nc x 1)

                % Style Transfer
                X_hat = hall_coeff*hall_img_scaled(:) + (1 - hall_coeff)*X_hat;
                
                % Robust Aggregation
                X_tilda = irls1(X_hat, ...
                               style_patch, ...
                               pca_out, ...
                               patch_sizes(j),r,...
                               IRLS_itr, ...
                               size(content_pyramid{i}), ...
                               sub_sampling_gaps(j), R);
         
                % Content Fusion
                mask = repmat(1.5*seg_mask_pyramid{i}(:)/max(seg_mask_pyramid{i}(:)),3,1);
                X_hat = (X_tilda + double(mask).*double(reshape(content_pyramid{i},[],1)))./(mask + ones(size(mask)));
                
                % Color Transfer
                X_hat = imhistmatch(reshape(X_hat, ...
                    size(content_pyramid{i})), style_pyramid{i});
                
                % Denoising
                X_hat = RF(X_hat, sigma_s, sigma_r);
                
            end
        end
        
        % Scale Up
        if (i~=L_max)
            X_hat = reshape(X_hat, size(content_pyramid{i}));
            X_hat = imresize(X_hat,2);
        end
        
    end
    
    output = X_hat;    
end
