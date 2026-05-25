import pytest
import math
from unittest.mock import MagicMock, patch
from agent import Agent, State

def make_mock_graph(nodes_and_neighbors: dict):
    """Создаёт мок-граф с заданными node_neighbors."""
    g = MagicMock()
    g.node_neighbors = nodes_and_neighbors
    return g


def make_mock_simulator(snapshots=None):
    sim = MagicMock()
    sim.snapshots = snapshots or []
    return sim


def make_snapshot(node_agents: dict):
    """
    node_agents: {node_id: [Agent, ...]}
    """
    return {"tick": 0, "node_occupants": node_agents}


def make_dashboard(nodes_and_neighbors=None, snapshots=None, layout_override=None):
    """
    Создаёт SimulationDashboard с замоканным nx.spring_layout.
    layout_override: кастомный dict позиций {node: (x, y)}
    """
    if nodes_and_neighbors is None:
        nodes_and_neighbors = {0: [1], 1: [0]}

    graph = make_mock_graph(nodes_and_neighbors)
    simulator = make_mock_simulator(snapshots)

    fixed_pos = layout_override or {n: (float(n), 0.0) for n in nodes_and_neighbors}

    with patch("dashboard.nx.spring_layout", return_value=fixed_pos):
        from dashboard import SimulationDashboard
        db = SimulationDashboard(simulator, graph)

    return db, fixed_pos


class TestInit:
    def test_simulator_stored(self):
        db, _ = make_dashboard()
        assert db.simulator is not None

    def test_graph_stored(self):
        db, _ = make_dashboard()
        assert db.graph is not None

    def test_pos_built_for_all_nodes(self):
        nodes = {0: [1, 2], 1: [0], 2: [0]}
        db, _ = make_dashboard(nodes_and_neighbors=nodes)
        assert set(db.pos.keys()) == {0, 1, 2}

    def test_pos_values_are_tuples_of_floats(self):
        db, _ = make_dashboard()
        for node, coords in db.pos.items():
            assert len(coords) == 2
            assert isinstance(coords[0], float)
            assert isinstance(coords[1], float)


class TestBuildLayout:
    def test_spring_layout_called_once(self):
        with patch("dashboard.nx.spring_layout") as mock_layout:
            mock_layout.return_value = {0: (0.0, 0.0), 1: (1.0, 0.0)}
            from importlib import reload
            import dashboard
            graph = make_mock_graph({0: [1], 1: [0]})
            sim = make_mock_simulator()
            dashboard.SimulationDashboard(sim, graph)
            mock_layout.assert_called_once()

    def test_edges_added_to_networkx_graph(self):
        """Все рёбра из node_neighbors должны попасть в nx.Graph."""
        captured = {}

        def fake_spring_layout(G, **kwargs):
            captured["edges"] = list(G.edges())
            return {n: (float(n), 0.0) for n in G.nodes()}

        nodes = {0: [1], 1: [0, 2], 2: [1]}
        graph = make_mock_graph(nodes)
        sim = make_mock_simulator()

        with patch("dashboard.nx.spring_layout", side_effect=fake_spring_layout):
            from dashboard import SimulationDashboard
            SimulationDashboard(sim, graph)

        edge_set = {frozenset(e) for e in captured["edges"]}
        assert frozenset({0, 1}) in edge_set
        assert frozenset({1, 2}) in edge_set

class TestGetAgentPositions:
    def setup_method(self):
        self.db, _ = make_dashboard()

    def test_returns_correct_number_of_positions(self):
        positions = self.db._get_agent_positions(0.0, 0.0, 4)
        assert len(positions) == 4

    def test_single_agent_at_offset(self):
        positions = self.db._get_agent_positions(0.0, 0.0, 1, radius=0.05)
        x, y = positions[0]
        assert abs(x - 0.05) < 1e-9
        assert abs(y - 0.0) < 1e-9

    def test_two_agents_are_opposite(self):
        positions = self.db._get_agent_positions(0.0, 0.0, 2, radius=1.0)
        x0, y0 = positions[0]
        x1, y1 = positions[1]
        assert abs(x0 + x1) < 1e-9
        assert abs(y0 + y1) < 1e-9

    def test_positions_lie_on_circle(self):
        cx, cy, r = 1.0, 2.0, 0.1
        positions = self.db._get_agent_positions(cx, cy, 6, radius=r)
        for x, y in positions:
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            assert abs(dist - r) < 1e-9

    def test_center_offset_applied(self):
        positions = self.db._get_agent_positions(3.0, 4.0, 1, radius=0.0)
        x, y = positions[0]
        assert abs(x - 3.0) < 1e-9
        assert abs(y - 4.0) < 1e-9

    def test_returns_list_of_tuples(self):
        positions = self.db._get_agent_positions(0.0, 0.0, 3)
        assert isinstance(positions, list)
        for item in positions:
            assert isinstance(item, tuple)
            assert len(item) == 2


class TestBuildFrame:
    def setup_method(self):
        nodes = {0: [1], 1: [0]}
        pos = {0: (0.0, 0.0), 1: (1.0, 0.0)}
        self.db, _ = make_dashboard(nodes_and_neighbors=nodes, layout_override=pos)

    def make_snapshot_with_agents(self):
        a1 = Agent(State.spreader, 0)
        a2 = Agent(State.ignorant, 1)
        return make_snapshot({0: [a1], 1: [a2]})

    def test_returns_go_frame(self):
        import plotly.graph_objects as go
        snap = self.make_snapshot_with_agents()
        frame = self.db._build_frame(snap)
        assert isinstance(frame, go.Frame)

    def test_frame_has_two_traces(self):
        snap = self.make_snapshot_with_agents()
        frame = self.db._build_frame(snap)
        assert len(frame.data) == 2

    def test_first_trace_is_edges(self):
        import plotly.graph_objects as go
        snap = self.make_snapshot_with_agents()
        frame = self.db._build_frame(snap)
        assert isinstance(frame.data[0], go.Scatter)
        assert frame.data[0].mode == "lines"

    def test_second_trace_is_agents(self):
        import plotly.graph_objects as go
        snap = self.make_snapshot_with_agents()
        frame = self.db._build_frame(snap)
        assert isinstance(frame.data[1], go.Scatter)
        assert frame.data[1].mode == "markers"

    def test_agent_colors_match_state(self):
        from dashboard import COLOR_MAP
        snap = self.make_snapshot_with_agents()
        frame = self.db._build_frame(snap)
        colors = list(frame.data[1].marker.color)
        assert COLOR_MAP[State.spreader] in colors
        assert COLOR_MAP[State.ignorant] in colors

    def test_empty_snapshot_produces_no_agents(self):
        snap = make_snapshot({0: [], 1: []})
        frame = self.db._build_frame(snap)
        assert len(frame.data[1].x) == 0

    def test_edge_trace_contains_none_separators(self):
        # Рёбра разделяются None для разрыва линий в plotly
        snap = self.make_snapshot_with_agents()
        frame = self.db._build_frame(snap)
        edge_x = list(frame.data[0].x)
        assert None in edge_x


class TestColorMap:
    def test_all_states_in_color_map(self):
        from dashboard import COLOR_MAP
        for state in State:
            assert state in COLOR_MAP

    def test_color_values_are_strings(self):
        from dashboard import COLOR_MAP
        for state, color in COLOR_MAP.items():
            assert isinstance(color, str)


class TestRun:
    def test_run_creates_figure_from_snapshots(self):
        a1 = Agent(State.spreader, 0)
        snap = make_snapshot({0: [a1], 1: []})
        nodes = {0: [1], 1: [0]}
        pos = {0: (0.0, 0.0), 1: (1.0, 0.0)}
        db, _ = make_dashboard(
            nodes_and_neighbors=nodes,
            snapshots=[snap],
            layout_override=pos
        )
        with patch("dashboard.Dash") as MockDash:
            mock_app = MagicMock()
            MockDash.return_value = mock_app
            mock_app.run = MagicMock()
            db.run()
            mock_app.run.assert_called_once()

    def test_run_builds_one_frame_per_snapshot(self):
        a1 = Agent(State.spreader, 0)
        snaps = [make_snapshot({0: [a1], 1: []})] * 3
        nodes = {0: [1], 1: [0]}
        pos = {0: (0.0, 0.0), 1: (1.0, 0.0)}
        db, _ = make_dashboard(
            nodes_and_neighbors=nodes,
            snapshots=snaps,
            layout_override=pos
        )
        frames = [db._build_frame(s) for s in db.simulator.snapshots]
        assert len(frames) == 3