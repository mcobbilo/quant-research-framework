import os
import ast
import json
import networkx as nx
from datetime import datetime


class GraphMemoryGenerator:
    def __init__(self, root_dir, research_dir):
        self.root_dir = root_dir
        self.research_dir = research_dir
        self.memory_dir = os.path.join(self.root_dir, "memory")
        self.obsidian_dir = os.path.join(self.root_dir, "obsidian_vault")
        self.graph = nx.DiGraph()
        self.concepts = [
            "VIX Scaling",
            "Kelly Criterion",
            "Regime Detection",
            "GMM",
            "TFT",
            "Non-Leveraged",
            "Counter-Cyclical",
            "Recurrent Memory",
            "VIX > 40",
            "MOVE > 130",
            "Dead Zone",
            "Stubbing",
            "Execution Safety",
        ]

    def scan_python_files(self):
        print(f"[Graphifier] Scanning Python files in {self.root_dir}...")
        for root, _, files in os.walk(self.root_dir):
            if "venv" in root or "__pycache__" in root:
                continue
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    self._parse_ast(path)

    def _parse_ast(self, file_path):
        rel_path = os.path.relpath(file_path, self.root_dir)
        self.graph.add_node(rel_path, type="file", category="code")

        with open(file_path, "r") as f:
            try:
                tree = ast.parse(f.read())
            except:
                return

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                node_id = f"{rel_path}::{node.name}"
                self.graph.add_node(node_id, type="class", category="code")
                self.graph.add_edge(rel_path, node_id, relation="contains")
                doc = ast.get_docstring(node)
                if doc:
                    self._link_doc_concepts(node_id, doc)
            elif isinstance(node, ast.FunctionDef):
                node_id = f"{rel_path}::{node.name}"
                self.graph.add_node(node_id, type="function", category="code")
                self.graph.add_edge(rel_path, node_id, relation="contains")
                doc = ast.get_docstring(node)
                if doc:
                    self._link_doc_concepts(node_id, doc)

    def scan_research_dir(self):
        print(f"[Graphifier] Scanning Research papers in {self.research_dir}...")
        for root, _, files in os.walk(self.research_dir):
            for file in files:
                if file.endswith((".md", ".txt")):
                    path = os.path.join(root, file)
                    rel_path = os.path.relpath(path, self.research_dir)

                    # Exclusion filter for noise
                    if any(
                        x in rel_path for x in ["personaplex", "tmp_", "build/", "venv"]
                    ):
                        continue

                    node_id = f"research://{rel_path}"
                    try:
                        self.graph.add_node(node_id, type="paper", category="research")
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            self._link_doc_concepts(node_id, content)
                    except Exception as e:
                        print(f"[Graphifier] Skipping {rel_path} due to error: {e}")

    def _link_doc_concepts(self, parent_node, text):
        for concept in self.concepts:
            if concept.lower() in text.lower():
                concept_node = f"concept://{concept}"
                if not self.graph.has_node(concept_node):
                    self.graph.add_node(concept_node, type="concept", category="domain")
                self.graph.add_edge(parent_node, concept_node, relation="references")

    def scan_memory_dir(self):
        print(f"[Graphifier] Scanning Ground-Truth Memory in {self.memory_dir}...")
        if not os.path.exists(self.memory_dir):
            return

        for root, _, files in os.walk(self.memory_dir):
            for file in files:
                if file.endswith(".md"):
                    path = os.path.join(root, file)
                    rel_path = os.path.relpath(path, self.memory_dir)
                    node_id = f"memory://{rel_path}"
                    try:
                        # Ground Truth files have higher weight in the graph
                        self.graph.add_node(
                            node_id, type="memory", category="ground_truth", weight=2.0
                        )
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                            self._link_doc_concepts(node_id, content)
                    except Exception as e:
                        print(f"[Graphifier] Skipping {rel_path} due to error: {e}")

    def scan_obsidian_vault(self):
        print(f"[Graphifier] Scanning Obsidian Vault in {self.obsidian_dir}...")
        if not os.path.exists(self.obsidian_dir):
            return

        for root, _, files in os.walk(self.obsidian_dir):
            for file in files:
                if file.endswith(".md"):
                    path = os.path.join(root, file)
                    rel_path = os.path.relpath(path, self.obsidian_dir)
                    node_id = f"obsidian://{rel_path}"
                    try:
                        self.graph.add_node(
                            node_id, type="doctrine", category="ground_truth"
                        )
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                            self._link_doc_concepts(node_id, content)
                    except Exception as e:
                        print(f"[Graphifier] Skipping {rel_path} due to error: {e}")

    def synthesize(self):
        # Calculate centrality
        degree_centrality = nx.degree_centrality(self.graph)
        nx.set_node_attributes(self.graph, degree_centrality, "centrality")

        # Save as JSON
        data = nx.node_link_data(self.graph)
        output_path = os.path.join(self.root_dir, "memory_graph.json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Graphifier] Exported knowledge graph to {output_path}")

        # Generate Human-Readable report
        report_path = os.path.join(self.root_dir, "MEMORY_MAP.md")
        with open(report_path, "w") as f:
            f.write("# Memory Map: Structural Repository Insights\n\n")
            f.write(
                f"**Generated At**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write("## God Nodes (Highest Centrality)\n")
            sorted_nodes = sorted(
                self.graph.nodes(data=True),
                key=lambda x: x[1].get("centrality", 0),
                reverse=True,
            )
            for node, attr in sorted_nodes[:10]:
                f.write(
                    f"- **{node}** (Type: {attr.get('type')}, Score: {attr.get('centrality'):.3f})\n"
                )

            f.write("\n## Concept Alignment\n")
            for concept in self.concepts:
                node_id = f"concept://{concept}"
                if self.graph.has_node(node_id):
                    refs = [u for u, v in self.graph.in_edges(node_id)]
                    f.write(f"### {concept}\n")
                    for ref in refs:
                        f.write(f"- {ref}\n")
        print(f"[Graphifier] Exported structural report to {report_path}")


if __name__ == "__main__":
    root = "/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework"
    research = "/Users/milocobb/Desktop/Recent Swarm Papers"
    gen = GraphMemoryGenerator(root, research)
    gen.scan_python_files()
    gen.scan_research_dir()
    gen.scan_memory_dir()
    gen.scan_obsidian_vault()
    gen.synthesize()
