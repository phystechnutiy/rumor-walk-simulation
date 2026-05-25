from dash import Dash, html, dcc
from agent import State
import networkx as nx
import plotly.graph_objects as go
from simulator import Simulator
from graph import Graph
import math

COLOR_MAP = {
    State.ignorant: "blue",
    State.spreader: "red",
    State.stifler: "gray",
    State.cooperator: "green",
}

class SimulationDashboard:
    """Interactive dashboard for visualizing rumor spreading simulation."""

    def __init__(self, simulator: Simulator, graph: Graph) -> None:
        """
        Initialize the dashboard.
        Args:
            simulator (Simulator): Simulator instance with snapshots.
            graph (Graph): Graph instance used in the simulation.
        """
        self.simulator = simulator
        self.graph = graph
        self.pos = self._build_layout()

    def _build_layout(self) -> dict[int, tuple[float, float]]:
        """
        Build spring layout for graph nodes.
        Returns:
            dict[int, tuple[float, float]]: Mapping from node to (x, y) coordinates.
        """
        G = nx.Graph()
        for node, neighbors in self.graph.node_neighbors.items():
            G.add_node(node)
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        return nx.spring_layout(G)

    def _get_agent_positions(self, cx: float, cy: float, n_agents: int, radius: float = 0.05) -> list[tuple[float, float]]:
        """
        Distribute agents around a node center in a circle.
        Args:
            cx (float): X coordinate of the node center.
            cy (float): Y coordinate of the node center.
            n_agents (int): Number of agents on the node.
            radius (float): Radius of the circle around the node.
        Returns:
            list[tuple[float, float]]: List of (x, y) positions for each agent.
        """
        return [
            (cx + radius * math.cos(2 * math.pi * i / n_agents),
             cy + radius * math.sin(2 * math.pi * i / n_agents))
            for i in range(n_agents)
        ]

    def _build_frame(self, snapshot: dict) -> go.Frame:
        """
        Build a single plotly frame for a given simulation snapshot.
        Args:
            snapshot (dict): Snapshot containing tick and node_occupants.
        Returns:
            go.Frame: Plotly frame with edges and agents.
        """
        edge_x, edge_y = [], []
        for node, neighbors in self.graph.node_neighbors.items():
            for neighbor in neighbors:
                x0, y0 = self.pos[node]
                x1, y1 = self.pos[neighbor]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

        agent_x, agent_y, agent_colors = [], [], []
        for node, agents in snapshot["node_occupants"].items():
            cx, cy = self.pos[node]
            positions = self._get_agent_positions(cx, cy, len(agents))
            for agent, (x, y) in zip(agents, positions):
                agent_x.append(x)
                agent_y.append(y)
                agent_colors.append(COLOR_MAP[agent.state])

        return go.Frame(data=[
            go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="lightgray", width=1)),
            go.Scatter(x=agent_x, y=agent_y, mode="markers", marker=dict(color=agent_colors, size=10)),
        ])

    def run(self) -> None:
        """Launch the Dash application in the browser."""
        frames = [self._build_frame(s) for s in self.simulator.snapshots]
        fig = go.Figure(
            data=frames[0].data,
            frames=frames,
            layout=go.Layout(
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(
                        label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=300, redraw=True), fromcurrent=True)]
                    )]
                )],
                sliders=[dict(
                    steps=[dict(method="animate", args=[[str(i)]], label=str(i))
                           for i in range(len(frames))],
                    currentvalue=dict(prefix="Tick: ")
                )]
            )
        )
        app = Dash(__name__)
        app.layout = html.Div([dcc.Graph(figure=fig, style={"height": "90vh"})])
        app.run()