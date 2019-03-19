function [] = linear_partitioning_visualization(tree_node, space_dims, lim, axes)
    % function [] = linear_partitioning_visualization(itree_node, space_dims)
    %
    % Visualize 3 dimensional space partitioning by a binary tree.
    %
    % Args:
    %   tree_node  ... root of the binary tree that partitions the space into regions.
    %   space_dims ... dimensions of the space being split. A 3x2 matrix
    %                  where the ith row represents the ith dimension. The
    %                  first column represents the lower bound and the second 
    %                  column the upper bound.
    %   axes       ... axes on which to plot
    %
    % Returns:
    %   unit/void
    
    % base case - node does not exist.
    if isnan(tree_node.split_val) || tree_node.level >= lim
        return
    else
        % Draw divising plane.
        draw_div_plane(tree_node.split_dim, tree_node.split_val, space_dims, axes);
        
        % Compute bounds of region to be split recursively.
        space_dims_nxt1 = space_dims;
        space_dims_nxt1(tree_node.split_dim, 2) = tree_node.split_val;
        % Recursive call for left subtree.
        linear_partitioning_visualization(tree_node.l, space_dims_nxt1, lim, axes);
        
        % Compute bounds of region to be split recursively.
        space_dims_nxt2 = space_dims;
        space_dims_nxt2(tree_node.split_dim, 1) = tree_node.split_val;
        % Recursive call for right subtree.
        linear_partitioning_visualization(tree_node.r, space_dims_nxt2, lim, axes);
        
    end
end
