"""Lightweight GraphRAG implementation using NetworkX.

Provides a local knowledge graph for agents to store and query
entities, relationships, and semantic embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any

import networkx as nx

from .state import CompressedVector, GraphEdge, GraphNode


@dataclass
class GraphQueryResult:
    """Result of a graph query."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    paths: list[list[str]] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


class GraphRAG:
    """Thread-safe knowledge graph with semantic search capabilities.

    Uses NetworkX for graph operations and supports:
    - Entity/relationship storage
    - Vector similarity search (via embeddings)
    - Multi-hop reasoning paths
    - Subgraph extraction for context
    """

    def __init__(self, embedding_dim: int = 384):
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._embedding_dim = embedding_dim
        self._node_embeddings: dict[str, CompressedVector] = {}
        self._lock = RLock()
        self._node_index: dict[str, set[str]] = {}  # type -> node_ids

    def add_node(self, node: GraphNode) -> str:
        """Add or update a node in the graph."""
        with self._lock:
            self._graph.add_node(node.node_id, **node.model_dump())

            # Update type index
            if node.node_type not in self._node_index:
                self._node_index[node.node_type] = set()
            self._node_index[node.node_type].add(node.node_id)

            # Store embedding if present
            if node.embedding_ref:
                # In production, this would fetch from vector store
                pass

            return node.node_id

    def add_edge(self, edge: GraphEdge) -> str:
        """Add or update an edge in the graph."""
        with self._lock:
            self._graph.add_edge(
                edge.source_id, edge.target_id, key=edge.edge_id, **edge.model_dump()
            )
            return edge.edge_id

    def get_node(self, node_id: str) -> GraphNode | None:
        """Retrieve a node by ID."""
        with self._lock:
            if node_id not in self._graph:
                return None
            data = self._graph.nodes[node_id]
            return GraphNode(**data)

    def get_edge(
        self, source_id: str, target_id: str, edge_id: str
    ) -> GraphEdge | None:
        """Retrieve a specific edge."""
        with self._lock:
            if not self._graph.has_edge(source_id, target_id):
                return None
            edge_data = self._graph.get_edge_data(source_id, target_id)
            if edge_id in edge_data:
                return GraphEdge(**edge_data[edge_id])
            return None

    def get_neighbors(
        self, node_id: str, relationship: str | None = None, direction: str = "out"
    ) -> list[tuple[GraphNode, GraphEdge]]:
        """Get neighboring nodes with connecting edges."""
        with self._lock:
            if node_id not in self._graph:
                return []

            results = []
            if direction in ("out", "both"):
                for _, target, key, data in self._graph.out_edges(
                    node_id, keys=True, data=True
                ):
                    if relationship is None or data.get("relationship") == relationship:
                        target_node = self.get_node(target)
                        if target_node:
                            results.append((target_node, GraphEdge(**data)))

            if direction in ("in", "both"):
                for source, _, key, data in self._graph.in_edges(
                    node_id, keys=True, data=True
                ):
                    if relationship is None or data.get("relationship") == relationship:
                        source_node = self.get_node(source)
                        if source_node:
                            results.append((source_node, GraphEdge(**data)))

            return results

    def find_nodes_by_type(self, node_type: str) -> list[GraphNode]:
        """Find all nodes of a specific type."""
        with self._lock:
            node_ids = self._node_index.get(node_type, set())
            nodes: list[GraphNode] = []
            for nid in node_ids:
                node = self.get_node(nid)
                if node is not None:
                    nodes.append(node)
            return nodes

    def find_nodes_by_property(
        self, property_key: str, property_value: Any
    ) -> list[GraphNode]:
        """Find nodes matching a property value."""
        with self._lock:
            results = []
            for node_id, data in self._graph.nodes(data=True):
                if data.get(property_key) == property_value:
                    results.append(GraphNode(**data))
            return results

    def semantic_search(
        self,
        query_vector: CompressedVector,
        top_k: int = 10,
        node_types: list[str] | None = None,
    ) -> list[tuple[GraphNode, float]]:
        """Search nodes by vector similarity.

        Note: In production, this would use a proper vector database.
        This is a mock implementation using cosine similarity.
        """
        with self._lock:
            import numpy as np

            query_arr = np.array(query_vector.values)
            query_norm = np.linalg.norm(query_arr)
            if query_norm == 0:
                return []

            candidates = []
            node_ids = set()

            if node_types:
                for nt in node_types:
                    node_ids.update(self._node_index.get(nt, set()))
            else:
                node_ids = set(self._graph.nodes())

            for node_id in node_ids:
                node = self.get_node(node_id)
                if node and node.embedding_ref:
                    # Mock: would fetch actual embedding from vector store
                    # For now, generate deterministic pseudo-embedding
                    import hashlib

                    seed = hashlib.md5(node_id.encode()).hexdigest()
                    np.random.seed(int(seed[:8], 16))
                    emb = np.random.randn(self._embedding_dim)
                    np.random.seed()

                    emb_norm = np.linalg.norm(emb)
                    if emb_norm > 0:
                        similarity = np.dot(query_arr, emb) / (query_norm * emb_norm)
                        candidates.append((node, float(similarity)))

            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:top_k]

    def find_paths(
        self,
        source: str,
        target: str,
        max_length: int = 3,
        relationship_filter: list[str] | None = None,
    ) -> list[list[str]]:
        """Find all paths between two nodes up to max_length."""
        with self._lock:
            if source not in self._graph or target not in self._graph:
                return []

            try:
                paths = list(
                    nx.all_simple_paths(self._graph, source, target, cutoff=max_length)
                )

                if relationship_filter:
                    filtered = []
                    for path in paths:
                        valid = True
                        for i in range(len(path) - 1):
                            edge_data = self._graph.get_edge_data(path[i], path[i + 1])
                            if edge_data:
                                rels = [
                                    d.get("relationship") for d in edge_data.values()
                                ]
                                if not any(r in relationship_filter for r in rels):
                                    valid = False
                                    break
                            else:
                                valid = False
                                break
                        if valid:
                            filtered.append(path)
                    return filtered

                return paths
            except nx.NetworkXNoPath:
                return []

    def extract_subgraph(
        self, center_node: str, radius: int = 2, max_nodes: int = 50
    ) -> GraphQueryResult:
        """Extract a subgraph around a center node for context."""
        with self._lock:
            if center_node not in self._graph:
                return GraphQueryResult(nodes=[], edges=[])

            # Get nodes within radius
            subgraph_nodes: set[str] = set()
            current_layer: set[str] = {center_node}

            for _ in range(radius):
                next_layer: set[str] = set()
                for node in current_layer:
                    subgraph_nodes.add(node)
                    next_layer.update(self._graph.successors(node))
                    next_layer.update(self._graph.predecessors(node))
                current_layer = next_layer - subgraph_nodes

            # Limit size
            if len(subgraph_nodes) > max_nodes:
                subgraph_nodes = set(list(subgraph_nodes)[:max_nodes])

            # Extract nodes and edges
            nodes: list[GraphNode] = []
            edges = []

            for node_id in subgraph_nodes:
                retrieved_node = self.get_node(node_id)
                if retrieved_node is not None:
                    nodes.append(retrieved_node)

            for u, v, key, data in self._graph.edges(keys=True, data=True):
                if u in subgraph_nodes and v in subgraph_nodes:
                    edges.append(GraphEdge(**data))

            return GraphQueryResult(nodes=nodes, edges=edges)

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        with self._lock:
            return {
                "node_count": self._graph.number_of_nodes(),
                "edge_count": self._graph.number_of_edges(),
                "node_types": {k: len(v) for k, v in self._node_index.items()},
                "density": (
                    nx.density(self._graph) if self._graph.number_of_nodes() > 0 else 0
                ),
                "is_dag": nx.is_directed_acyclic_graph(self._graph),
            }

    def clear(self) -> None:
        """Clear the entire graph."""
        with self._lock:
            self._graph.clear()
            self._node_embeddings.clear()
            self._node_index.clear()


class GraphRAGManager:
    """Manages multiple GraphRAG instances per simulation epoch."""

    def __init__(self):
        self._graphs: dict[str, GraphRAG] = {}
        self._lock = RLock()

    def get_or_create(self, epoch_id: str, embedding_dim: int = 384) -> GraphRAG:
        """Get or create a GraphRAG for an epoch."""
        with self._lock:
            if epoch_id not in self._graphs:
                self._graphs[epoch_id] = GraphRAG(embedding_dim)
            return self._graphs[epoch_id]

    def remove(self, epoch_id: str) -> bool:
        """Remove a graph for an epoch."""
        with self._lock:
            if epoch_id in self._graphs:
                self._graphs[epoch_id].clear()
                del self._graphs[epoch_id]
                return True
            return False

    def list_epochs(self) -> list[str]:
        """List all active epochs."""
        with self._lock:
            return list(self._graphs.keys())


# Global instance
graph_rag_manager = GraphRAGManager()
