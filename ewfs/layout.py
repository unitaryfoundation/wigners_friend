import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

class Node:
    """
    A simple class to represent a node in a binary tree.
    Each node has a unique ID, a left child, and a right child.
    """
    def __init__(self, node_id):
        self.id = node_id
        self.left = None
        self.right = None

    def __repr__(self):
        return f"Node({self.id})"

class MockCouplingMap:
    """A simple class to represent the connectivity of a quantum device."""
    def __init__(self, adj_list):
        self._adj = adj_list.copy() # Use a copy to avoid modifying the original dict

    def neighbors(self, node):
        """Returns the neighbors of a given node."""
        return self._adj.get(node, [])

    def add_node(self, node, neighbors_list=None):
        """
        Adds a new node to the coupling map, optionally connecting it to existing nodes.

        If a neighbor in the list does not exist, it will be created.

        Args:
            node (int): The ID of the new node to add.
            neighbors_list (list, optional): A list of nodes to connect to the new node.
        """
        if neighbors_list is None:
            neighbors_list = []

        # Add the new node to the adjacency list if it doesn't exist
        if node not in self._adj:
            self._adj[node] = []
        
        # Connect the new node to each of its specified neighbors
        for neighbor in neighbors_list:
            # Add bidirectional connection
            if neighbor not in self._adj[node]:
                self._adj[node].append(neighbor)
            
            # Ensure the neighbor node exists and connect it back to the new node
            if neighbor not in self._adj:
                self._adj[neighbor] = [node]
            elif node not in self._adj[neighbor]:
                self._adj[neighbor].append(node)
    
    def get_adj(self):
        """Returns the underlying adjacency list."""
        return self._adj

def build_complete_binary_tree(depth):
    """
    Builds a complete binary tree of a given depth.
    Nodes are numbered starting from 1.
    Returns the root node of the tree.
    """
    if depth < 0:
        return None

    nodes = [Node(i) for i in range(1, 2**(depth + 1))]
    for i in range(len(nodes)):
        left_child_idx = 2 * i + 1
        right_child_idx = 2 * i + 2
        if left_child_idx < len(nodes):
            nodes[i].left = nodes[left_child_idx]
        if right_child_idx < len(nodes):
            nodes[i].right = nodes[right_child_idx]
    return nodes[0]

def get_all_nodes(root):
    """
    Performs a traversal to find all nodes in the tree.
    """
    if not root:
        return []
    all_nodes = []
    stack = [root]
    while stack:
        node = stack.pop()
        all_nodes.append(node)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return all_nodes


def get_leaf_nodes(root):
    """
    Performs a traversal to find all leaf nodes in the tree.
    Returns a list of Node objects.
    """
    if not root:
        return []
    leaves = []
    stack = [root]
    while stack:
        node = stack.pop()
        if not node.left and not node.right:
            leaves.append(node)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return leaves

def find_path(root, target_node_id, path=[]):
    """
    Finds the path from the root to a node with the given ID.
    Returns the path as a list of nodes.
    """
    if root is None:
        return None

    current_path = list(path)
    current_path.append(root)

    if root.id == target_node_id:
        return current_path

    if root.left:
        left_path = find_path(root.left, target_node_id, current_path)
        if left_path:
            return left_path

    if root.right:
        right_path = find_path(root.right, target_node_id, current_path)
        if right_path:
            return right_path

    return None

def find_lca(root, node1_id, node2_id):
    """
    Finds the Least Common Ancestor (LCA) of two nodes.
    """
    path1 = find_path(root, node1_id)
    path2 = find_path(root, node2_id)

    if not path1 or not path2:
        return None

    lca = None
    for n1, n2 in zip(path1, path2):
        if n1.id == n2.id:
            lca = n1
        else:
            break
    return lca

def calculate_coverage(root, node1_id, node2_id):
    """
    Calculates the "coverage" of a pair of leaf nodes.
    Coverage is the set of unique nodes in the paths from
    each leaf to their least common ancestor.
    """
    lca = find_lca(root, node1_id, node2_id)
    if not lca:
        return 0, set(), None

    path1 = find_path(lca, node1_id)
    path2 = find_path(lca, node2_id)
    
    # Combine paths and remove duplicates to get the coverage set
    coverage_path_nodes = set(path1 + path2)
    return coverage_path_nodes, lca

def greedy_pair_selection(root, alpha):
    """
    Implements the greedy algorithm to select pairs of leaf nodes
    that balance maximizing coverage and minimizing the number of pairs.
    """
    all_nodes_list = get_all_nodes(root)
    total_node_count = len(all_nodes_list)
    if total_node_count == 0:
        return [], set()

    leaves = get_leaf_nodes(root)
    if len(leaves) < 2:
        return [], set()

    # Pre-calculate all possible pair coverages
    potential_pairs = {}
    for leaf1, leaf2 in combinations(leaves, 2):
        coverage_set, _ = calculate_coverage(root, leaf1.id, leaf2.id)
        if coverage_set:
            potential_pairs[(leaf1.id, leaf2.id)] = {node.id for node in coverage_set}

    # --- Greedy Algorithm ---
    selected_pairs = []
    covered_node_ids = set()
    
    k = 0
    # Objective function F(k) = k + alpha * (UncoveredRatio)
    current_score = k + alpha * ((total_node_count - len(covered_node_ids)) / total_node_count)

    while potential_pairs:
        best_pair_to_add = None
        max_newly_covered_count = -1

        # Find the pair that covers the most new nodes
        for pair, coverage_ids in potential_pairs.items():
            newly_covered_nodes = coverage_ids - covered_node_ids
            if len(newly_covered_nodes) > max_newly_covered_count:
                max_newly_covered_count = len(newly_covered_nodes)
                best_pair_to_add = pair

        if best_pair_to_add is None or max_newly_covered_count == 0:
            break # No more progress can be made

        # Calculate the score if we add this best pair
        next_k = k + 1
        next_covered_count = len(covered_node_ids) + max_newly_covered_count
        next_score = next_k + alpha * ((total_node_count - next_covered_count) / total_node_count)

        # Stopping condition: if the new score is not better, stop.
        if next_score >= current_score:
            print("Stopping: Adding more pairs does not improve the objective function score.")
            break
        
        # Update state
        current_score = next_score
        k = next_k
        selected_pairs.append(best_pair_to_add)
        covered_node_ids.update(potential_pairs[best_pair_to_add])
        del potential_pairs[best_pair_to_add]
        
    return selected_pairs, covered_node_ids

def find_optimal_k_pairs(root, k):
    """
    Finds the 'k' pairs of leaf nodes that are predicted to maximize coverage
    using a greedy approach. This is an implementation of the Maximum Coverage problem.
    """
    leaves = get_leaf_nodes(root)
    if len(leaves) < 2 or k == 0:
        return [], set()

    # Pre-calculate all possible pair coverages
    potential_pairs = {}
    for leaf1, leaf2 in combinations(leaves, 2):
        coverage_set, _ = calculate_coverage(root, leaf1.id, leaf2.id)
        if coverage_set:
            potential_pairs[(leaf1.id, leaf2.id)] = {node.id for node in coverage_set}

    # --- Greedy selection for k pairs ---
    selected_pairs = []
    covered_node_ids = set()

    for _ in range(k):
        if not potential_pairs:
            break # Stop if we run out of pairs before reaching k

        best_pair_to_add = None
        max_newly_covered_count = -1

        # Find the pair that covers the most new nodes
        for pair, coverage_ids in potential_pairs.items():
            newly_covered_nodes = coverage_ids - covered_node_ids
            if len(newly_covered_nodes) > max_newly_covered_count:
                max_newly_covered_count = len(newly_covered_nodes)
                best_pair_to_add = pair
        
        if best_pair_to_add is None:
            break # No pair can add new nodes

        # Add the best pair to our solution
        selected_pairs.append(best_pair_to_add)
        covered_node_ids.update(potential_pairs[best_pair_to_add])
        del potential_pairs[best_pair_to_add] # Remove from consideration

    return selected_pairs, covered_node_ids

def tree_to_adj_list(root):
    """
    Converts a Node-based tree to an adjacency list dictionary.
    """
    if not root:
        return {}
    adj = {}
    q = [root]
    visited = {root.id}
    while q:
        node = q.pop(0)
        if node.id not in adj:
            adj[node.id] = []
        if node.left:
            adj[node.id].append(node.left.id)
            if node.left.id not in adj:
                adj[node.left.id] = []
            adj[node.left.id].append(node.id)
            if node.left.id not in visited:
                q.append(node.left)
                visited.add(node.left.id)
        if node.right:
            adj[node.id].append(node.right.id)
            if node.right.id not in adj:
                adj[node.right.id] = []
            adj[node.right.id].append(node.id)
            if node.right.id not in visited:
                q.append(node.right)
                visited.add(node.right.id)
    return adj

def create_coupling_map_from_selection(root, selected_pairs):
    """
    Creates a MockCouplingMap from a binary tree and adds new nodes for selected pairs.
    Returns the coupling map, a list of all node IDs, and a list of the new pair node IDs.
    """
    # 1. Convert the entire tree to a base adjacency list.
    adj = tree_to_adj_list(root)
    if not adj:
        return MockCouplingMap({}), [], []

    pair_nodes = []
    # 2. Add new "pair" nodes and their edges.
    if selected_pairs:
        new_node_id_start = max(adj.keys()) + 1
        for i, pair in enumerate(selected_pairs):
            new_node_id = new_node_id_start + i
            leaf1_id, leaf2_id = pair
            pair_nodes.append(new_node_id)

            # Add the new node and its connections to the leaves.
            adj[new_node_id] = [leaf1_id, leaf2_id]

            # Add connections from the leaves back to the new node.
            adj[leaf1_id].append(new_node_id)
            adj[leaf2_id].append(new_node_id)

    # 3. Get all nodes from the final adjacency list.
    all_final_nodes = list(adj.keys())

    # 4. Instantiate and return the final coupling map object and lists.
    return MockCouplingMap(adj), all_final_nodes, pair_nodes

def visualize_pair_selection_solution(root, selected_pairs, final_covered_ids, title):
    """
    Visualizes the result of the greedy selection on the original tree.
    """
    G = nx.Graph()
    pos = {}
    labels = {}

    def add_edges_and_positions(node, x=0, y=0, level_height=1, level_width=1):
        if node is not None:
            G.add_node(node.id)
            pos[node.id] = (x, -y)
            labels[node.id] = str(node.id)
            if node.left:
                G.add_edge(node.id, node.left.id)
                add_edges_and_positions(node.left, x - level_width / 2, y + level_height, level_height, level_width / 2)
            if node.right:
                G.add_edge(node.id, node.right.id)
                add_edges_and_positions(node.right, x + level_width / 2, y + level_height, level_height, level_width / 2)

    add_edges_and_positions(root)

    # --- Determine node and edge colors ---
    selected_leaf_ids = {item for t in selected_pairs for item in t}
    lca_ids = {find_lca(root, p[0], p[1]).id for p in selected_pairs}

    node_colors = []
    for node_id in G.nodes():
        if node_id in selected_leaf_ids:
            node_colors.append('tomato')  # Selected leaf
        elif node_id in lca_ids:
            node_colors.append('gold') # LCA of a selected pair
        elif node_id in final_covered_ids:
            node_colors.append('lightblue') # Covered by a path
        else:
            node_colors.append('lightgray') # Uncovered

    # Color edges that are part of any selected coverage path
    coverage_edges = set()
    for pair in selected_pairs:
        lca = find_lca(root, pair[0], pair[1])
        path1, _ = calculate_coverage(lca, pair[0], pair[0])
        path2, _ = calculate_coverage(lca, pair[1], pair[1])
        
        path1_ids = {n.id for n in path1}
        path2_ids = {n.id for n in path2}

        for u, v in G.edges():
            if (u in path1_ids and v in path1_ids) or \
               (u in path2_ids and v in path2_ids):
                coverage_edges.add(tuple(sorted((u, v))))

    edge_colors = ['blue' if tuple(sorted(edge)) in coverage_edges else 'gray' for edge in G.edges()]

    plt.figure(figsize=(16, 10))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=800, node_color=node_colors, edge_color=edge_colors, width=2.0, font_size=10)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # --- Configuration ---
    TREE_DEPTH = 2
    NUM_PAIRS_TO_SELECT = 1

    # --- Script Execution ---
    root_node = build_complete_binary_tree(TREE_DEPTH)
    all_nodes = get_all_nodes(root_node)
    
    print(f"Tree Depth: {TREE_DEPTH}")
    print(f"Total nodes in tree: {len(all_nodes)}")

    # --- Method 2: Find optimal coverage for a fixed k ---
    print("\n" + "="*50)
    print(f"METHOD 2: Maximizing Coverage with k={NUM_PAIRS_TO_SELECT} Pairs")
    print("="*50)
    
    selected_pairs_k, ids_k = find_optimal_k_pairs(root_node, NUM_PAIRS_TO_SELECT)
    ratio_k = len(ids_k) / len(all_nodes) if all_nodes else 0
    
    print(f"Number of pairs selected: {len(selected_pairs_k)}")
    print(f"Selected pairs: {selected_pairs_k}")
    print(f"Total nodes covered: {len(ids_k)}")
    print(f"Coverage Ratio: {ratio_k:.2%}")

    # Create and show a sample of the coupling map for this method
    coupling_map_k, nodes, pair_nodes_k = create_coupling_map_from_selection(root_node, selected_pairs_k)
    print("\nCoupling map created for Method 2.")
    if pair_nodes_k:
        # Example: Show neighbors of the first new pair node
        first_new_node_k = pair_nodes_k[0]
        print(f"Example: Neighbors of new node {first_new_node_k}: {coupling_map_k.neighbors(first_new_node_k)}")

    if selected_pairs_k:
        visualize_pair_selection_solution(root_node, selected_pairs_k, ids_k,
                                          title=f"Optimal Coverage for k={NUM_PAIRS_TO_SELECT} Pairs")

