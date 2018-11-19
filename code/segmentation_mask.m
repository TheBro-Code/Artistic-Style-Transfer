%% Segmentation function 

function mask = segmentation_mask(content,threshold,sigma1,sigma2)
    content = rgb2gray(content);
    edgy_content = edge(content,'log',threshold,sigma1);
    mask = imgaussfilt(double(edgy_content),sigma2);
end
