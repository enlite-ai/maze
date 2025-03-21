"""Contains tests for the PyG gnn based block"""
import pytest
import torch

from maze.perception.blocks.feed_forward.gnn_pyg import SUPPORTED_GNNS, GNNLayerPyG, _dummy_edge_index_factory, \
    GNNBlockPyG


def test_gnn_layer_init_unsupported_type():
    """Test that initialising GNNLayerPyG with an unsupported type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported GNN type"):
        GNNLayerPyG(in_features=16, out_features=32, bias=True, gnn_type="invalid")


def test_dummy_edge_index_factory():
    """
    Test the _dummy_edge_index_factory.
    """
    n_nodes = 5
    n_edges = 10

    shape = (2, n_edges)
    factory = _dummy_edge_index_factory(shape, n_nodes)
    edge_index = factory()

    assert edge_index.shape == (1, shape[0], shape[1])

    # All values should be in [0, n_nodes-1]
    assert (edge_index >= 0).all() and (edge_index < n_nodes).all()

@pytest.mark.parametrize("gnn_type", SUPPORTED_GNNS)
@pytest.mark.parametrize("batch_size", [None, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_gnn_layer_forward(gnn_type, batch_size, bias):
    """
    Test forward pass.
    """
    in_features = 8
    out_features = 16
    n_nodes = 5

    layer = GNNLayerPyG(in_features=in_features, out_features=out_features, bias=False, gnn_type=gnn_type)

    # Construct inputs
    if batch_size is None:
        x = torch.randn(n_nodes, in_features)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_attr = torch.randn(edge_index.shape[1])  # (E,) for GCN or even random
    else:
        x = torch.randn(batch_size, n_nodes, in_features)
        edge_index = []
        edge_attr = []
        for _ in range(batch_size):
            e_i = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
            e_a = torch.randn(e_i.shape[1])
            edge_index.append(e_i)
            edge_attr.append(e_a)

        edge_index = torch.stack(edge_index, dim=0)
        edge_attr = torch.stack(edge_attr, dim=0)

    out = layer(x, edge_index, edge_attr)

    if batch_size is None:
        assert out.shape == (n_nodes, out_features)
    else:
        assert out.shape == (batch_size, n_nodes, out_features)


@pytest.mark.parametrize("gnn_type", SUPPORTED_GNNS)
def test_gnn_block_forward(gnn_type: str):
    """
    Test the forward pass of GNNBlockPyG with dummy data.
    """

    batch_size = 2
    n_nodes = 4
    n_edges = 10
    in_features = 8
    hidden_features = [16, 16]

    block = GNNBlockPyG(
        in_keys=["node_feats", "edge_index", "edge_attr"],
        out_keys=["out"],
        in_shapes=[(n_nodes, in_features), (2, n_edges), (n_edges, )],
        hidden_features=hidden_features,
        non_lin="torch.nn.ReLU",
        bias=True,
        gnn_type=gnn_type,
    )

    node_feats = torch.randn(batch_size, n_nodes, in_features)

    # Edge index shape expected: (batch_size, 2, E)  =>  E = 10 here
    edge_index = []
    edge_attr = []
    for _ in range(batch_size):
        e_i = torch.randint(0, n_nodes, (2, n_edges))
        e_a = torch.randn((n_edges, 1))
        edge_index.append(e_i)
        edge_attr.append(e_a)

    edge_index = torch.stack(edge_index, dim=0)
    edge_attr = torch.stack(edge_attr, dim=0)

    block_input = {
        "node_feats": node_feats,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }

    output_dict = block(block_input)
    out = output_dict["out"]

    assert out.shape == (batch_size, n_nodes, hidden_features[-1])

    assert not torch.isnan(out).any()
