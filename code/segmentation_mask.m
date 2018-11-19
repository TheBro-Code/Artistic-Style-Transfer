function mask = segmentation_mask(content,threshold,sigma1,sigma2)
    edge_out = edge(rgb2gray(content), 'log',threshold,sigma1);
    mask = imgaussfilt(double(edge_out), sigma2);
    imagesc(mask)
end
