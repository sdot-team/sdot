from collections import defaultdict

def inspect_graph_readable(fn):

    node_ids = {}
    counter = [1]

    def get_id(obj):
        oid = id(obj)
        if oid not in node_ids:
            node_ids[oid] = counter[0]
            counter[0] += 1
        return node_ids[oid]

    root_id = get_id(fn)

    print(f"Node {root_id}: {type(fn).__name__}")

    grouped = defaultdict(list)

    for parent, out_idx in fn.next_functions:
        if parent is not None:
            grouped[id(parent)].append((parent, out_idx))

    for i, group in enumerate(grouped.values()):

        parent = group[0][0]
        pid = get_id(parent)

        outputs = [f"output {o}" for _, o in group]

        connector = "├──" if i < len(grouped)-1 else "└──"

        print(f"{connector} uses {', '.join(outputs)} from Node {pid}: {type(parent).__name__}")

        for gp, idx in parent.next_functions:
            if gp is not None:
                gid = get_id(gp)
                print(f"    └── parent output {idx} → Node {gid}: {type(gp).__name__}")