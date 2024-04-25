import torch
import torch.optim as optim
from ScaleHyperprior import ScaleHyperprior_chengres as model

def configure_optimizers(net):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=1e-4,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=1e-3,
    )
    return optimizer, aux_optimizer



# Path to the checkpoint file
checkpoint_path = "ScaleHyperprior_chengres3FalseTrue_checkpoint_best.pth.tar"
#

model = model()
optimizer, aux_optimizer = configure_optimizers(model)


# Load the checkpoint
checkpoint = torch.load(checkpoint_path)

# last_epoch = checkpoint["epoch"] + 1
# best_loss = checkpoint["best_loss"]
# net.load_state_dict(checkpoint["state_dict"])
# optimizer.load_state_dict(checkpoint["optimizer"])
# aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
# lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
print(checkpoint['state_dict'].keys())  
#rename state_dict and keep all the other keys
new_checkpoint = {'state_dict': {}}
for key in checkpoint.keys():
    # If the key is 'state_dict', process its inner keys
    if key == "state_dict":
        for key2 in checkpoint[key].keys():
            # Remove the '_orig_mod.' prefix
            new_key = key2.replace('_orig_mod.', '')
            # Assign the modified key to the new_checkpoint
            new_checkpoint['state_dict'][new_key] = checkpoint[key][key2]
    else:
        # Copy all other keys as they are
        new_checkpoint[key] = checkpoint[key]

# Now, new_checkpoint contains the modified 'state_dict' and all other keys are unchanged
# print(new_checkpoint.keys())
# print(new_checkpoint['state_dict'].keys())  

#save checkpoint
# torch.save(new_checkpoint, 'ScaleHyperprior_EfficientV2_LN3FalseTrue_checkpoint_best_new.pth.tar')
