import networkx as nx


def get_seed_rep_max(G: nx.Graph, rep_vals: dict, num_seeds: int = 1) -> list:
    """Choose seeds based on the absolute accumulated reputation."""
    # Choose new seeds
    new_seeds = sorted(rep_vals.items(), key=lambda x: x[1], reverse=True)
    new_seeds = [k for k,_ in new_seeds]
    
    cur_seeds = []
    selected_seeds = 0

    while selected_seeds < num_seeds:
        for s in new_seeds:
            if s in G:
                cur_seeds.append(s)
                selected_seeds += 1
                if selected_seeds == num_seeds:
                    break
        if selected_seeds != num_seeds:
            print('Cannot find enough seeds!')
            return None
    return cur_seeds
    


