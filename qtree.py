from dtree import DNode
import numpy as np
from timeit import default_timer
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import argparse

class QTree:
    """Q-table + tile coding implemented by decision tree"""
    
    def __init__(self, lower, upper, split, action_size, adaptive=False, p=100):
        """Create regular tile, implemented with tree structure
        Parameters
        ----------
        lower: array-like
            Lower bounds of grid for each dimension
        upper: array-like
            Upper bounds of grid for each dimension
        split: int
            number of cells for each dimension
            TODO: same value for every dimension
        action_size: int
            number of admissible actions
        adaptive : bool
            use adaptive tile code or not
        p: int 
            global constant parameter used to decide if split occurs or not
        """
        self.action_size = action_size
        self.state_sizes = len(lower)
        self.adaptive = adaptive
        self.p = p

        self.qtree = self.regular_qtree(lower, upper, split, action_size)

        self.counter_update = 0
        self.counter_value = []

    def initialize_subtree(self, new_node):
        middle = (new_node.lower + new_node.upper) / 2
        new_node.subtrees = []
        for d in range(self.state_sizes):
            node = DNode(middle[d], d)
            # left
            node.left = DNode(2 * new_node.value + 1)
            node.left.q = np.zeros(self.action_size)
            node.left.lower = new_node.lower
            node.left.upper = new_node.upper.copy()
            node.left.upper[d] = middle[d]
            node.left.abs_delta_q = np.inf
            # right
            node.right = DNode(2 * new_node.value + 2)
            node.right.q = np.zeros(self.action_size)
            node.right.lower = new_node.lower.copy()
            node.right.lower[d] = middle[d]
            node.right.upper = new_node.upper
            node.right.abs_delta_q = np.inf

            new_node.subtrees.append(node)

    def regular_qtree(self, lower, upper, split, action_size):
        MAXHEIGHT = int(np.log2(split) - 1)
        ACTIONSIZE = action_size

        init_lower = np.array(lower)
        init_upper = np.array(upper)
        init_dim = 0
        init_middle = (init_lower + init_upper) / 2
        init_idx = 0
        init_height = 0

        DIM = len(lower)
        assert len(lower) == len(upper) # state vector dimension should match

        root = DNode(init_middle[init_dim], init_dim)

        def reggrid_nd_q(idx, lower, upper, cur_height, n_dim):

            # stop condition for each dimension
            if cur_height > MAXHEIGHT:
                # terminal ?
                if n_dim == DIM - 1:
                    root[idx] = DNode(idx)
                    root[idx].q = np.zeros(ACTIONSIZE)

                    # keep bound info for growing adaptive tree
                    if self.adaptive:
                        root[idx].lower = lower
                        root[idx].upper = upper
                        root[idx].abs_delta_q = np.inf
                        self.initialize_subtree(root[idx])
                    return
                # grow tree for next dimension
                else:
                    cur_height = 0
                    n_dim += 1

            middle_ndim = (lower[n_dim] + upper[n_dim]) / 2
            # create node if "root" is not really a real root node of tree
            if idx > 0:
                root[idx] = DNode(middle_ndim, n_dim)

            left_upper = upper.copy()
            left_upper[n_dim] = middle_ndim
            reggrid_nd_q(2*idx+1, lower, left_upper, cur_height + 1, n_dim)

            right_lower = lower.copy()
            right_lower[n_dim] = middle_ndim
            reggrid_nd_q(2*idx+2, right_lower, upper, cur_height + 1, n_dim)

        reggrid_nd_q(init_idx, init_lower, init_upper, init_height, init_dim)
        return root

    def search_spacial(self, root, test_array, return_root=False): 
        """binary search on decision tree"""
        # Base Cases: root is null or key is present at root 
        if root is None or root.dim is None: 
            # returns q-value array, and its index of the node in the tree
            if return_root:
                return root
            else:
                return root.q, root.value

        # Key is greater than root's key 
        if test_array[root.dim] < root.value: 
            return self.search_spacial(root.left, test_array, return_root)
        else:
            # Key is smaller than root's key 
            return self.search_spacial(root.right, test_array, return_root)
        
    def get(self, state, action=None, return_index=True):
        """Get Q-value for given <state, action> pair.
        
        Parameters
        ----------
        state: array_like
            array representing the state in the original continuous space
        action: int or None
            Index or label of action.  if None, the corresponding q-array is returned
        return_index: bool
            returns index of the node if true
            
        Returns
        -------
        value: float
            Q-value of given <state, action> pair
        idx: int
            index of the node in which the <state, action> pair falls
        """
    
        # TODO: parse state tree to get q-value array
        #qarray, idx = self.search_spacial(self.qtree, state)
        node = self.search_spacial(self.qtree, state, return_root=True)
        qarray = node.q
        if action is None:
            return qarray, node
        else:
            # TODO: get q-value by accessing array
            q_value = qarray[action]
            if return_index:
                return q_value, node#, idx
            else:
                return q_value
            
    def update(self, state, action, value, alpha=0.1):
        """Soft-update Q-value for given <state, action> pair to value.
        
        Instead of overwriting Q(state, action) with value, perform soft-update:
            Q(state, action) = alpha * value + (1.0 - alpha) * Q(state, action)
        
        Parameters
        ----------
        state : array_like
            Vector representing the state in the original continuous space.
        action : int
            Index of desired action.
        value : float
            Desired Q-value for <state, action> pair.
        alpha : float
            Update factor to perform soft-update, in [0.0, 1.0] range.
        """
        # get current value and its reference
        #q_value_current, idx = self.get(state, action)
        q_value_current, node = self.get(state, action)
        # update the value based on observed value
        self.counter_update += 1
        t_start = default_timer()
        #self.qtree[idx].q[action] = alpha * value + (1.0 - alpha) * q_value_current
        node.q[action] = alpha * value + (1.0 - alpha) * q_value_current
        t_end = default_timer()
        t_diff = t_end - t_start
        self.counter_value.append(t_diff)

        if self.counter_update % 1000 == 0:
            t_diff_mean = sum(self.counter_value) / 1000
            #print(f'action update at idx: {t_diff_mean:.8f} sec / 1000 update')
        if self.adaptive:
            # update subtile weight
            for d in range(self.state_sizes):
                #subtree = self.qtree[idx].subtrees[d]
                subtree = node.subtrees[d]
                subnode = self.search_spacial(subtree, state, return_root=True)

                #subtree[(idx_sub+1) % 2 + 1].q[action] = alpha * value + (1.0 - alpha) * qarray[action]
                subnode.q[action] = alpha * value + (1.0 - alpha) * subnode.q[action]

            # split criterion ("WHEN TO SPLIT")
            abs_delta_q = abs(value - q_value_current)

            if abs_delta_q < node.abs_delta_q:
                # upate lowest delta q
                node.abs_delta_q = abs_delta_q
                # no split this time
                self.u = 0
            else:
                # split "potential" goes up
                self.u += 1
            
            # check "potential" exceeds the threshold ("WHERE TO SPLIT")
            if self.u > self.p:
                #print('----- SUBTREE SPLIT! -----')
                
                # Value criterion: find maximal subweights difference
                #idxmax = np.argmax([sum((tree.left.q - tree.right.q)**2) for tree in self.qtree[idx].subtrees])
                idxmax = np.argmax([sum((tree.left.q - tree.right.q)**2) for tree in node.subtrees])

                # replace terminal node with the selected subtree
                #self.qtree[idx] = self.qtree[idx].subtrees[idxmax]
                subnode = node.subtrees[idxmax]
                node.left = subnode.left
                node.right = subnode.right
                node.dim = subnode.dim
                node.value = subnode.value

                # initialize children as 
                #self.qtree[idx].left.abs_delta_q = np.inf
                node.left.abs_delta_q = np.inf
                #self.initialize_subtree(self.qtree[idx].left)
                self.initialize_subtree(node.left)
                #self.qtree[idx].right.abs_delta_q = np.inf
                node.right.abs_delta_q = np.inf
                #self.initialize_subtree(self.qtree[idx].right)
                self.initialize_subtree(node.right)

                #print(self.qtree)
                #print('--------------------------')
                # init u
                self.u = 0

    def visualize_split(self):
        leaf_count = self.qtree.leaf_count
        # pick rectangle info
        rectparam = []
        for leaf in self.qtree.leaves:
            lw = np.array(leaf.lower)
            up = np.array(leaf.upper)
            
            widthheight = up - lw
            
            idx = leaf.value
            rectparam.append((tuple(lw), tuple(widthheight), idx))
            
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111)

        #p.set_transform(ax.transAxes)
        #p.set_clip_on(False)
        count = 0
        for rp in rectparam:
            #if count > 20:    
            rect = patches.Rectangle(rp[0], *rp[1], edgecolor='orange', linewidth=1, alpha=0.2)
            left, bottom = rp[0]
            width, height = rp[1]
            right = left + width
            top = bottom + height
            ax.text(0.5 * (left + right), 0.5 * (bottom + top), rp[2],
                    horizontalalignment='center',
                    verticalalignment='center',
                    )
            ax.add_patch(rect) 
            
            count += 1
            
        ax.set_xlim((-1.2,0.6))
        ax.set_ylim((-0.07,0.07))
        plt.show()
        

# test code
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adaptive', action='store_true')
    args = parser.parse_args()

    low = [-1.0, -5.0]
    high = [1.0, 5.0]
    # Test with some sample values
    samples = [(-1.2 , -5.1 ),
            (-0.75,  3.25),
            (-0.5 ,  0.0 ),
            ( 0.25, -1.9 ),
            ( 0.15, -1.75),
            ( 0.75,  2.5 ),
            ( 0.7 , -3.7 ),
            ( 1.0 ,  5.0 )]    

    # Test with a sample Q-table
    tq = QTree(low, high, 4, 2, args.adaptive)
    s1 = 3; s2 = 4; a = 0; q = 1.0
    # check value at sample = s1, action = a
    print("[GET]    Q({}, {}) = {}".format(samples[s1], a, tq.get(samples[s1], a)))  
     # update value for sample with some common tile(s)
    print("[UPDATE] Q({}, {}) = {}".format(samples[s2], a, q))
    tq.update(samples[s2], a, q) 
    # check value again, should be slightly updated
    print("[GET]    Q({}, {}) = {}".format(samples[s1], a, tq.get(samples[s1], a)))  