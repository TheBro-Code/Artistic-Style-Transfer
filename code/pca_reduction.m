function pca_out = pca_reduction(S,k)

training_set = double(S);
mean_data = mean(training_set,2);
training_set = training_set - repmat(mean_data,[1,size(training_set,2)]);

L = training_set*training_set';
[W,E] = eig(L);
V = normc(W);
proj_mat = V(:,size(V,2)-k+1:size(V,2));
S_eig = proj_mat'*training_set;
pca_out = {proj_mat,S_eig,mean_data};

end
