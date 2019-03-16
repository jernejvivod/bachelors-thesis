class It_node:
    def __init__(self, l, r, split_attr, split_val, level, mass_comp=0):
        self.l = l  # left subtree
        self.r = r  # right subree
        self.split_attr = split_attr  # split attribute/dimension
        self.split_val = split_val    # split value
        self.level = level            # node level
        self.mass = 0                 # node mass (number of examples in region)
        self.mass_comp = mass_comp    # node mass computed at same time as tree is being built (compare with mass computed with traversals)

    # to_string: return a string encoding some information about the node
    def to_string(self):
        return "split_attr={0}, split_val={1}, level={2}, mass={3}".format(self.split_attr,\
                                                                 self.split_val,\
                                                                 self.level,
                                                                 self.mass)

