function [Id,out] = nearest_neighbour(S,R)
    [Id,out] = knnsearch(S',R');
end