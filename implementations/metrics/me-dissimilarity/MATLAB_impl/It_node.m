classdef It_node
   % Class representing a node in an itree
   properties
       l
       r
       split_dim
       split_val
       level
   end
   methods
       % Constructor
       function obj = It_node(l, r, split_dim, split_val, level)
           obj.l = l;
           obj.r = r;
           obj.split_dim = split_dim;
           obj.split_val = split_val;
           obj.level = level;
       end
   end
end