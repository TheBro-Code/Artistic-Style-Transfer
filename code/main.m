clear; clc;

%% Inputs

% reading content and style image

Content_img = imread('../images/content/elsa.jpg');
style_img = imread('../images/styles/pencil.jpg');

% Segmentation Mask
mask = segmentation_mask(Content_img, 0.03, 1 , 2);

%

style_new = imhistmatch(Content_img, style_img);
imagesc(style_new);


