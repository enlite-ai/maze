""" Contains inference blocks. """
import os
import pickle
from typing import Union, List, Dict, Optional, Sequence, Tuple

import networkx as nx
import torch
from matplotlib import pyplot as plt
from torch import nn as nn

from maze.core.annotations import override
from maze.perception.blocks.base import PerceptionBlock


class InferenceBlock(PerceptionBlock):
    """An inference block combining multiple perception blocks into one prediction module.
            Conditions on using the InferenceBlock object:
                1. All keys of the perception_blocks dictionary have to be unique
                2. All out_keys used when creating the blocks have to be unique
                3. All block keys in the perception_blocks dict have to sub-strings of all their corresponding out_keys
                4. The given in_keys should be a subset of the inputs of the computational graph
                5. The given out_keys should be a subset of the outputs of the computational graph

    :param in_keys: Keys identifying the input tensors.
    :param out_keys: Keys identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param perception_blocks: Dictionary of perception blocks.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]],
                 perception_blocks: Dict[str, PerceptionBlock]):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes)
        self._test_condition(perception_blocks, self.in_keys, self.out_keys)
        self.block_keys = list(perception_blocks.keys())
        self.perception_dict = perception_blocks
        self.perception_blocks = nn.ModuleDict(perception_blocks)
        # Copy the in_keys
        self.execution_plan = self._build_execution_graph(computed_keys=self.in_keys[:], execution_plan=dict())

    @classmethod
    def _test_condition(cls, perception_blocks: Dict[str, PerceptionBlock], in_keys: List[str], out_keys: List[str]):
        """Test the defined conditions

        :param perception_blocks: Dictionary of perception blocks.
        :param in_keys: The in_keys of the perception block
        :param out_keys: The out_keys of the perception block
        """
        # 1. All keys of the perception_blocks dictionary have to be unique
        assert len(set(perception_blocks.keys())) == len(list(perception_blocks.keys()))

        # 2. All out_keys used when creating the blocks have to be unique
        all_out_keys = sum([block.out_keys for block in perception_blocks.values()], [])
        assert len(set(all_out_keys)) == len(list(all_out_keys))

        # 3. All block keys in the perception_blocks dict have to sub-strings of all their corresponding out_keys
        for block_key, block in perception_blocks.items():
            for out_key in block.out_keys:
                assert block_key in out_key, f'out_key ({out_key}) is not a superset of the block_key ' \
                                             f'({block_key}) it belongs to.'

        # Test the structure of the graph
        all_in_keys = sum([block.in_keys for block in perception_blocks.values()], [])
        all_out_keys = sum([block.out_keys for block in perception_blocks.values()], [])

        # 4. The inputs to the computational graph should be a subset of the given in_keys
        graph_in_keys = set(all_in_keys) - set(all_out_keys)
        assert all([in_key in in_keys for in_key in graph_in_keys]), \
            f'in_keys of the network: {set(all_in_keys) - set(all_out_keys)}, vs specified in_keys: {set(in_keys)}'

        # 5. The given out_keys should be a subset of the outputs of the computational graph
        assert all([out_key in all_out_keys for out_key in out_keys]), \
            f'Out_keys of the network: {set(all_out_keys) - set(all_in_keys)}, vs specified out_keys: {set(out_keys)}'

    @override(PerceptionBlock)
    def forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.base.PerceptionBlock` interface
        """
        assert all([key in block_input.keys() for key in self.in_keys]), \
            f'The specified in_keys {self.in_keys} should be a subset of the observations: {block_input.keys()}'

        tmp_dict: Dict[str, torch.Tensor] = dict()

        for step_idx in range(len(self.execution_plan.keys())):
            argument_dict = {**tmp_dict, **block_input}
            step_block_keys = self.execution_plan[step_idx]

            for block_key in step_block_keys:
                block_output = self.perception_blocks[block_key](argument_dict)

                # update tensor dictionary
                tmp_dict.update(block_output)

        assert all([key in tmp_dict for key in self.out_keys]), 'All out_keys should be computed at this point'
        # compile output dictionary
        out_dict = dict()
        for out_key in self.out_keys:
            out_dict[out_key] = tmp_dict[out_key]

        return out_dict

    def _build_execution_graph(self, computed_keys: List[str],
                               execution_plan: Dict[int, List[str]]) -> Dict[int, List[str]]:
        """Build the execution graph for computing the forward path through the network.

        :param computed_keys: The named tensors computed up until this point.
        :param execution_plan: The order in which to process the blocks in the forward pass
        :return: The order in which to process the blocks in the forward pass
        """
        computed_blocks = sum(execution_plan.values(), [])
        current_step_blocks = []
        current_step_out_keys = []
        for block_key, block in self.perception_blocks.items():
            if block_key not in computed_blocks:
                if all([in_key in computed_keys for in_key in block.in_keys]):
                    current_step_blocks.append(block_key)
                    current_step_out_keys += block.out_keys

        execution_plan[len(execution_plan.keys())] = current_step_blocks
        computed_keys += current_step_out_keys

        if all([out_key in computed_keys for out_key in self.out_keys]):
            return execution_plan
        else:
            return self._build_execution_graph(computed_keys, execution_plan)


class InferenceGraph:
    """ Models a perception module inference graph.
            Conditions on using the InferenceGraph object:
                1. All keys of the perception_blocks dictionary have to be unique
                2. All out_keys used when creating the blocks have to be unique
                3. All out_keys of a given block have to be sub-strings of the blocks key in the perception_blocks dict

    :param inference_block: An inference perception block to build the graph for.
    """

    def __init__(self, inference_block: InferenceBlock):
        self.perception_blocks = inference_block.perception_blocks
        self.node_graph = nx.DiGraph()

        # build inference graph
        out_keys = inference_block.out_keys
        self.in_key_shapes = dict()
        self._total_num_of_params = self._get_num_of_parameters()
        self._build_inference_graph(out_keys=out_keys)

    def save(self, name: str, save_path: str) -> None:
        """Construct the network and save it as a pdf.

        :param name: The name of the network to be drawn (used in the tile only).
        :param save_path: The path the figure should be saved.
        """
        self._draw(name=name, figure_size=(18, 12))
        full_save_path = os.path.join(save_path, name + '.pdf')
        if not os.path.exists(full_save_path):
            print(f'Graphical depiction of the model \'{name}\' saved at: {os.path.abspath(full_save_path)}')
        plt.savefig(full_save_path, transparent=True)
        # If specifies pickle the figure and store it to be loaded and added to tensorboard at a later point in time.
        full_save_path = full_save_path.replace('.pdf', '.figure.pkl')
        pickle.dump(plt.gcf(), open(full_save_path, 'wb'))

        plt.clf()
        plt.close()

    def show(self, name: str, block_execution: bool) -> None:
        """Construct the graph and show it.

        :param name: The name of the network to be drawn (used in the tile only).
        :param block_execution: Specify whether the execution should be blocked.
        """
        self._draw(name=name, figure_size=None)
        plt.show(block=block_execution)

    def _draw(self, name: str, figure_size: Optional[Tuple[int, int]]) -> None:
        """Draws the inference graph using matplotlib.

        :param name: The name of the network to be drawn (used in the tile only)
        :param figure_size: The figure size to be drawn. If None, the size is inferred (does not work well for saving
            though.
        """

        fig = plt.figure(figsize=figure_size)
        plt.clf()
        renderer = fig.canvas.get_renderer()

        # get relevant node labels
        labels = nx.get_node_attributes(self.node_graph, "label")
        is_block = nx.get_node_attributes(self.node_graph, "is_block")
        num_params = nx.get_node_attributes(self.node_graph, 'num_params')
        pos = nx.nx_agraph.graphviz_layout(self.node_graph, prog='dot')

        max_params_per_node = max(num_params.values())

        # draw nodes
        bbox_heights = dict()
        for node in self.node_graph.nodes():
            x, y = pos[node]
            if not is_block[node]:
                color = 'indianred'
                alpha = 0.15
            elif num_params[node] == 0:
                color = 'powderblue'
                alpha = 0.1
            else:
                color = 'lightskyblue'
                alpha = float(num_params[node]) / max_params_per_node * 0.3 + 0.1
            obj = plt.text(x, y, labels[node], picker=True, ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor=color, alpha=alpha))
            plt.plot(x, y, "bo", markersize=0)

            # get bbox height
            bbox_height = obj.get_window_extent(renderer).height
            bbox_heights[node] = bbox_height

        # draw edges
        for src_node, dst_node, attrs in self.node_graph.edges(data=True):
            src_x, src_y = pos[src_node]
            dst_x, dst_y = pos[dst_node]
            src_y -= (bbox_heights[src_node] // 4 * 3) + 2
            dst_y += (bbox_heights[dst_node] // 4 * 3) + 2
            plt.annotate("", xy=(dst_x, dst_y), xytext=(src_x, src_y), arrowprops=dict(arrowstyle="->"))

        plt.axis("off")
        title = f"Graphical depiction of \'{name}\'"
        title += f' with {self._total_num_of_params:,} parameters'
        plt.title(title)
        plt.tight_layout()

    def _get_num_of_parameters(self) -> int:
        """Calculates the total number of parameters in the model

        :return: The total number of parameters in the model
        """
        return sum(pp.numel() for block in self.perception_blocks.values() for pp in block.parameters())

    def _get_block_name_for_out_key(self, out_key: str) -> str:
        """Retrieve the block key for a given out_key

        :param out_key: The out_key of an unknown block
        :return: The block key corresponding to the out_key
        """
        for block_name, block in self.perception_blocks.items():
            if out_key in block.out_keys:
                return block_name

    def _build_inference_graph(self, out_keys: List[str], parent: Optional[str] = None) -> None:
        """Recursively compiles the perception inference graph.

        :param out_keys: The out keys to start depth search from.
        :param parent: The parent key we are coming from.
        """
        # iterate out keys
        for out_key in out_keys:

            # Retrieve the key (w.r.t. the perception_blocks dict) of the block the given output (out_key) was computed
            #   in
            block_key = self._get_block_name_for_out_key(out_key)

            # Add output node if not present in the graph
            created_output_node = False
            if out_key not in self.node_graph.nodes():
                if block_key in self.perception_blocks:
                    out_index = self.perception_blocks[block_key].out_keys.index(out_key)
                    shape = self.perception_blocks[block_key].out_shapes()[out_index]
                    label = f"{out_key}\nshapes: {shape}"
                else:
                    shape = self.in_key_shapes[out_key]
                    label = f"{out_key}\nshape: {shape}"

                self.node_graph.add_node(out_key, label=label, is_block=False, num_params=0)
                created_output_node = True

            # Add edge from parent to output node if parent is given (not network output)
            if parent is not None:
                self.node_graph.add_edge(out_key, parent)

            # Continue if the block key is in perception_blocks (that is if this is an input) and if the given
            #   output node is already in the node graph (that is if we join paths)
            if block_key in self.perception_blocks and created_output_node:
                # Add the block node to the graph
                curr_block, block_id = self._add_node_to_node_graph(block_key)
                # Connect the block node to the output node
                self.node_graph.add_edge(block_id, out_key)

                # collect in key shapes
                for i, in_key in enumerate(curr_block.in_keys):
                    self.in_key_shapes[in_key] = curr_block.in_shapes[i]

                self._build_inference_graph(out_keys=curr_block.in_keys, parent=block_id)

    def _add_node_to_node_graph(self, block_key: str) -> Tuple[nn.Module, str]:
        """Add a node representing a perception block to the graph with the correct label

        :param block_key: The block the label should be generated for
        :return: The current blocks and the corresponding block id
        """

        curr_block = self.perception_blocks[block_key]
        block_id = "{}\n({})".format(curr_block.__class__.__name__, block_key)
        num_of_block_params = sum(pp.numel() for pp in curr_block.parameters())

        label = str(curr_block).replace('\t', '')
        if num_of_block_params > 0:
            label += f'\n#{num_of_block_params:,} = {num_of_block_params / self._total_num_of_params * 100:.1f}%'

        self.node_graph.add_node(block_id, label=label, is_block=True, num_params=num_of_block_params,
                                 block_id=block_key)
        return curr_block, block_id
