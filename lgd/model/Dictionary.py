import torch

NODE_DICT = {
    'NO_NODE': 0, 'VIRTUAL': -1, 'MASKED': -2, 'NO_ATTR': -3
}

EDGE_DICT = {
    'NO_EDGE': 0, 'VIRTUAL': -1, 'MASKED': -2, 'NO_ATTR': -3, 'LOOP': -4  # self-loop of (i,i)
}


cond_stage_config_list = {
    "__is_first_stage__", "__is_unconditional__"  # TODO: and other ckpt path
}  # used for initialize condition encoder model

cond_stage_key_list = {
    'unconditional', 'masked_graph', 'prompt_graph', 'prefix', 'class_label', 'multi-modal', 'pe'
}  # condition types

# here 'prefix' are tunable vectors used in cross attention to finetune the generative model;
# different from prefix tuning for encoder

conditioning_key_list = {
    'concat', 'crossattn', 'hybrid', 'adm'
}  # implementation method to encode conditions


# sparse feature without virtual node and virtual edges: x_original, edge_attr_original, edge_index_original
# unmasked values (with virtual node and edges): x_unmasked, edge_attr_unmasked
