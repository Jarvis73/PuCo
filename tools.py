import torch
from sacred import Experiment


ex = Experiment("Tools", save_git_info=False, base_dir="./")


@ex.command(unobserved=True)
def print_ckpt(ckpt):
    state = torch.load(ckpt, map_location='cpu')
    if 'model_state' in state:
        state = state['model_state']
    elif 'state_dict' in state:
        state = state['state_dict']
    elif 'model' in state:
        state = state['model']

    max_name_length = max([len(x) for x in state])
    max_shape_length = max([len(str(x.shape)) for x in state.values()])
    pattern = "  {:<%ds}  {:<%ds}" % (max_name_length, max_shape_length)

    print_str = ""
    for k, v in state.items():
        print_str += pattern.format(k, str(list(v.shape))) + "\n"

    print(print_str)
