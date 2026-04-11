import json


def extract_subgraph(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    nodes = data["nodes"]
    edges = data["edges"]

    node_map = {n["id"]: n for n in nodes}

    # 1. Select Hub Nodes
    hub_types = ["concept", "memory", "doctrine"]
    hubs = [n for n in nodes if n.get("type") in hub_types]

    # 2. Select High Centrality Nodes
    top_files = sorted(
        [n for n in nodes if n.get("type") == "file"],
        key=lambda x: x.get("centrality", 0),
        reverse=True,
    )[:15]
    top_papers = sorted(
        [n for n in nodes if n.get("type") == "paper"],
        key=lambda x: x.get("centrality", 0),
        reverse=True,
    )[:10]

    selected_node_ids = set([n["id"] for n in hubs + top_files + top_papers])

    # 3. Find edges between selected nodes
    subgraph_edges = [
        e
        for e in edges
        if e["source"] in selected_node_ids and e["target"] in selected_node_ids
    ]

    # 4. For concepts, bring in some representative neighbors to show "what's connected"
    concepts = [n["id"] for n in nodes if n.get("type") == "concept"]
    for concept_id in concepts:
        concept_edges = [
            e for e in edges if e["source"] == concept_id or e["target"] == concept_id
        ]
        # Sample some neighbors that aren't already in our set
        neighbors = concept_edges[:5]
        for e in neighbors:
            selected_node_ids.add(e["source"])
            selected_node_ids.add(e["target"])
            subgraph_edges.append(e)

    # Deduplicate edges based on source, target, relation
    seen_edges = set()
    unique_edges = []
    for e in subgraph_edges:
        edge_key = (e["source"], e["target"], e.get("relation"))
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            unique_edges.append(e)

    # Finalize nodes based on unique edges
    final_node_ids = set()
    for e in unique_edges:
        final_node_ids.add(e["source"])
        final_node_ids.add(e["target"])

    final_nodes = [node_map[nid] for nid in final_node_ids if nid in node_map]

    # Create Mermaid syntax
    mermaid = ["graph TD"]
    # Node styles
    # concept: octagon, memory: box, code: rounded, research: parallelogram, doctrine: cylinder
    for n in final_nodes:
        nid = n["id"]
        # Clean ID for Mermaid (no spaces, special chars)
        safe_id = (
            nid.replace("://", "_")
            .replace(".", "_")
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace(">", "GT")
        )
        label = nid.split("/")[-1]

        ntype = n.get("type", "node")
        if ntype == "concept":
            mermaid.append(f"  {safe_id}{{{{{label}}}}}")
        elif ntype == "memory":
            mermaid.append(f"  {safe_id}[({label})]")  # cylinder-ish
        elif ntype == "file" or ntype == "function" or ntype == "class":
            mermaid.append(f'  {safe_id}["{label}"]')
        elif ntype == "paper":
            mermaid.append(f"  {safe_id}[/{label}/]")
        else:
            mermaid.append(f'  {safe_id}(["{label}"])')

    for e in unique_edges:
        source_id = (
            e["source"]
            .replace("://", "_")
            .replace(".", "_")
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace(">", "GT")
        )
        target_id = (
            e["target"]
            .replace("://", "_")
            .replace(".", "_")
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace(">", "GT")
        )
        relation = e.get("relation", "connected")
        mermaid.append(f'  {source_id} -- "{relation}" --> {target_id}')

    return "\n".join(mermaid)


if __name__ == "__main__":
    print(extract_subgraph("memory_graph.json"))
