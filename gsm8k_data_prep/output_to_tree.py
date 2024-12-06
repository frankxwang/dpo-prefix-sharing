import pickle as pkl

class RadixNode:
    def __init__(self, label=''):
        self.label = label  # Label on the edge to this node
        self.children = []
        self.is_end = False  # Is this node the end of a word

def insert(node, word):
    current = node
    while True:
        # Find the child whose label shares a common prefix with word
        for child in current.children:
            common_prefix_len = common_prefix_length(child.label, word)
            if common_prefix_len > 0:
                # There is a common prefix between the child's label and word
                if common_prefix_len == len(child.label):
                    # The child's label is fully matched
                    word = word[common_prefix_len:]
                    current = child
                    if len(word) == 0:
                        # The word is fully inserted
                        current.is_end = True
                        return
                    break  # Break out of for loop to continue with this child
                else:
                    # Need to split the child
                    # Create new child with the non-matching part of the child's label
                    new_child = RadixNode(child.label[common_prefix_len:])
                    new_child.children = child.children
                    new_child.is_end = child.is_end
                    # Update the child
                    child.label = child.label[:common_prefix_len]
                    child.children = [new_child]
                    child.is_end = False
                    # Now, if there is remaining part of word, add it as child
                    if len(word[common_prefix_len:]) > 0:
                        new_leaf = RadixNode(word[common_prefix_len:])
                        new_leaf.is_end = True
                        child.children.append(new_leaf)
                    else:
                        child.is_end = True
                    return
        else:
            # No matching child found, add new child
            new_child = RadixNode(word)
            new_child.is_end = True
            current.children.append(new_child)
            return

def common_prefix_length(s1, s2):
    # Return length of common prefix between s1 and s2
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return i
    return min_len

def print_tree(node, indent=''):
    if node.label != '':
        print(indent + node.label + (' [end]' if node.is_end else ''))
    for child in node.children:
        print_tree(child, indent + '  ')

if __name__ == '__main__':
    with open("output_60.pkl", "rb") as f:
        trajs = pkl.load(f)
    root = RadixNode()
    for traj in trajs:
        insert(root, traj)
    print_tree(root)