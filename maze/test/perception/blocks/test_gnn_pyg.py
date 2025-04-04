"""Contains tests for the PyG gnn based block"""
import pytest
import torch

from maze.perception.blocks.feed_forward.gnn_pyg import SUPPORTED_GNNS, GNNLayerPyG, _dummy_edge_index_factory, \
    GNNBlockPyG


def test_gnn_layer_init_unsupported_type():
    """Test that initialising GNNLayerPyG with an unsupported type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported GNN type"):
        GNNLayerPyG(in_features=16, out_features=32, gnn_type="invalid", gnn_kwargs=None)


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

    layer = GNNLayerPyG(in_features=in_features, out_features=out_features, gnn_type=gnn_type, gnn_kwargs={'bias': bias})

    # Construct inputs
    if batch_size is None:
        x = torch.randn(n_nodes, in_features)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_attr = torch.randn(edge_index.shape[1])  # (E,) for GCN
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
    Test the forward pass of GNNBlockPyG.
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
        gnn_type=gnn_type,
        gnn_kwargs=None
    )

    node_feats = torch.randn(batch_size, n_nodes, in_features)

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


@pytest.mark.parametrize("batch_size", [None, 3])
@pytest.mark.parametrize("edge_dim", [None, 1, 5])
@pytest.mark.parametrize("n_heads", [1, 2, 4])

def test_gnn_layer_forward_gat_with_edge_attrs(batch_size, edge_dim, n_heads):
    """
    Test GNNLayerPyG layer with gnn_type='gat'.
    """
    gnn_type = "gat"
    in_features = 6
    out_features = 8
    n_nodes = 5
    n_edges = 7

    layer = GNNLayerPyG(
        in_features=in_features,
        out_features=out_features,
        gnn_type=gnn_type,
        gnn_kwargs={"edge_dim": edge_dim, "heads": n_heads}
    )

    if edge_dim is None:
        edge_dim = 1

    if batch_size is None:
        x = torch.randn(n_nodes, in_features)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        edge_attr = torch.randn(n_edges, edge_dim)
    else:
        x = torch.randn(batch_size, n_nodes, in_features)
        edge_index_list = []
        edge_attr_list = []
        for _ in range(batch_size):
            e_i = torch.randint(0, n_nodes, (2, n_edges))
            e_a = torch.randn(n_edges, edge_dim)
            edge_index_list.append(e_i)
            edge_attr_list.append(e_a)
        edge_index = torch.stack(edge_index_list, dim=0)
        edge_attr = torch.stack(edge_attr_list, dim=0)

    out = layer(x, edge_index, edge_attr)

    if batch_size is None:
        assert out.shape == (n_nodes, out_features * n_heads)
    else:
        assert out.shape == (batch_size, n_nodes, out_features * n_heads)
    assert not torch.isnan(out).any(), "Output contains NaNs!"

@pytest.mark.parametrize("n_heads", [1, 2, 4])
@pytest.mark.parametrize("edge_dim", [2, 3])
@pytest.mark.parametrize("concat", [True, False])
def test_gnn_block_forward_gat_with_edge_attrs(n_heads: int, edge_dim: int, concat: bool):
    """
    Test GNNBlockPyG with gnn_type='gat'.
    """
    batch_size = 2
    n_nodes = 4
    n_edges = 5
    in_features = 8
    hidden_features = [16, 16]

    block = GNNBlockPyG(
        in_keys=["node_feats", "edge_index", "edge_attr"],
        out_keys=["out"],
        in_shapes=[(n_nodes, in_features), (2, n_edges), (n_edges, edge_dim)],
        hidden_features=hidden_features,
        non_lin="torch.nn.ReLU",
        gnn_type="gat",
        gnn_kwargs={"edge_dim": edge_dim, "heads": n_heads, "concat": concat}
    )

    node_feats = torch.randn(batch_size, n_nodes, in_features)
    edge_index_list = []
    edge_attr_list = []
    for _ in range(batch_size):
        e_i = torch.randint(0, n_nodes, (2, n_edges))
        e_a = torch.randn((n_edges, edge_dim))
        edge_index_list.append(e_i)
        edge_attr_list.append(e_a)

    edge_index = torch.stack(edge_index_list, dim=0)
    edge_attr = torch.stack(edge_attr_list, dim=0)

    block_input = {
        "node_feats": node_feats,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }

    output_dict = block(block_input)
    out = output_dict["out"]

    if 'heads' in block.gnn_kwargs and ('concat' not in block.gnn_kwargs or block.gnn_kwargs['concat']):
        expected_output_features = hidden_features[-1] * block.gnn_kwargs['heads']
    else:
        expected_output_features = hidden_features[-1]

    assert out.shape == (batch_size, n_nodes, expected_output_features), \
        f"Expected shape (B={batch_size}, N={n_nodes}, out_feats={expected_output_features}) but got {out.shape}"

    # Check for NaNs
    assert not torch.isnan(out).any(), "Output contains NaNs!"
