""" Unit tests for selection block. """

import torch

from maze.perception.blocks.general.multi_index_slicing import MultiIndexSlicingBlock


def test_multi_index_slicing_block_multiple_ids():
    """ perception test """

    mask = torch.rand([2, 1, 4])
    in_shape = (mask.shape[-1],)
    select_block = MultiIndexSlicingBlock(in_keys='mask', out_keys='selected', in_shapes=in_shape, select_dim=-1,
                                          select_idxs=[0, 2])

    selected = select_block({'mask': mask})
    assert selected['selected'].shape == torch.Size([2, 1, 2])


def test_multi_index_slicing_block_single():
    """ perception test """

    mask = torch.rand([2, 1, 4])
    in_shape = (mask.shape[-1],)
    select_block = MultiIndexSlicingBlock(in_keys='mask', out_keys='selected', in_shapes=in_shape, select_dim=-1,
                                          select_idxs=[0])

    selected = select_block({'mask': mask})
    assert selected['selected'].shape == torch.Size([2, 1, 1])


def test_multi_index_slicing_block_single_cuda():
    """ perception test """
    if torch.cuda.is_available():
        mask = torch.rand([2, 1, 4]).to('cuda')
        in_shape = (mask.shape[-1],)
        select_block = MultiIndexSlicingBlock(in_keys='mask', out_keys='selected', in_shapes=in_shape, select_dim=-1,
                                              select_idxs=[0])

        selected = select_block({'mask': mask})
        assert selected['selected'].shape == torch.Size([2, 1, 1])
        str(select_block)
