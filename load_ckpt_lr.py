import torch
import torch.optim as optim
from ScaleHyperprior import SwinTHyperprior_TIC as model

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
        lr=222e-4,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=1e-3,
    )
    return optimizer, aux_optimizer



# Path to the checkpoint file
checkpoint_path = "SwinTHyperprior_TIC1TrueFalse_checkpoint.pth.tar"
#
model = model()
optimizer, aux_optimizer = configure_optimizers(model)


# Load the checkpoint
checkpoint = torch.load(checkpoint_path)

optimizer.load_state_dict(checkpoint["optimizer"])

# Get the lr value from the checkpoint
lr = optimizer.param_groups[0]["lr"]

# Print the lr value
print("Learning rate:", lr)

#load epoch
epoch = checkpoint["epoch"]
print("epoch:", epoch)