import networkx as nx
from typing import List, Dict, Any

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def add_entity(self, entity: str, entity_type: str = "unknown", properties: Dict = None):
        self.graph.add_node(entity, type=entity_type, **(properties or {}))
    
    def add_relation(self, subj: str, rel: str, obj: str, evidence: str = None):
        self.graph.add_edge(subj, obj, relation=rel, evidence=evidence)
    
    def get_neighbors(self, entity: str) -> List[str]:
        return list(self.graph.neighbors(entity))
    
    def get_relations(self, entity: str) -> List[Dict]:
        edges = []
        for u, v, data in self.graph.edges(data=True):
            if u == entity or v == entity:
                edges.append({
                    "subject": u,
                    "relation": data.get("relation", "unknown"),
                    "object": v,
                    "evidence": data.get("evidence")
                })
        return edges
    
    def get_entity_properties(self, entity: str) -> Dict:
        return dict(self.graph.nodes.get(entity, {}))
    
    def search(self, query: str) -> List[str]:
        return [node for node in self.graph.nodes if query.lower() in node.lower()]
    
    def to_text_context(self) -> str:
        nodes = list(self.graph.nodes)
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append(f"{u} --{data.get('relation','rel')}--> {v}")
        return "Graph nodes: " + ", ".join(nodes) + "\nEdges: " + "; ".join(edges)