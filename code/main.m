clear; clc;
tic;

%% Inputs
 
% reading content and style image

content_img = imread('../images/content/lena.jpg');
style_img = imread('../images/styles/pencil.jpg');

% segmentation mask 

threshold = 0.03;
sigma_edge = 1;
sigma_blur = 3;
seg_mask = segmentation_mask(content_img,threshold,sigma_edge,sigma_blur);

% Number of resolution layers

L_max = 3;

% patch sizes

patch_sizes = [33, 21, 13, 9];

% sub_sampling gaps

sub_sampling_gaps = [28, 18, 8, 5];

% Number of IRLS iterations

IRLS_itr = 10;

% number of update iterations per patch-size

I_alg = 3;

% robust statistics value to use

r = 0.8;

%% Style Transfer

stylised_result = style_transfer(content_img, ...
                                 style_img, ...
                                 L_max, ...
                                 seg_mask, ...
                                 patch_sizes, ...
                                 sub_sampling_gaps(2), ...
                                 IRLS_itr,I_alg,r);

subplot(1,3,1), imshow(content_img);
subplot(1,3,2), imshow(style_img);
subplot(1,3,3), imshow(stylised_result);

%%

toc;


