import torch
from torch.utils import data

def train(flow, x_samples, context=None, num_epochs=5000, batch_size=50, lr=0.005, plot=True, plt_min=0, plt_max=10, validation = False, epochs_until_converge = 20):
    tensors = [x_samples]
    if context is not None:
        tensors.append(context)

    dataset = data.TensorDataset(*tensors)

    if validation:
        val_size = int(0.3* len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = data.DataLoader(train_set, batch_size=batch_size)
        val_loader = data.DataLoader(val_set, batch_size=batch_size)
    else:
        train_loader = data.DataLoader(dataset, batch_size=batch_size)
        val_set = []

    opt = torch.optim.Adam(flow.parameters(), lr=lr)

    best_val_loss = 0
    epochs_since_last_improvement = 0
    nb_improvements = 0
    epoch = 0
    while (epoch < num_epochs) and (epochs_since_last_improvement < epochs_until_converge):

        for sample_batch in train_loader:
            opt.zero_grad()
            loss = -flow.log_prob(*sample_batch).mean()
            loss.backward()
            opt.step()

        if validation:
            with torch.no_grad():
                val_loss = 0 
                for sample_batch in val_loader:
                    batch_loss = -flow.log_prob(*sample_batch).mean()
                    val_loss+=batch_loss.sum().item()
                # Take mean over all validation samples.
                val_loss = val_loss / len(val_loader)

            if epoch == 0 or best_val_loss > val_loss:
                best_val_loss = val_loss
                epochs_since_last_improvement = 0
                nb_improvements+=1
            else:
                epochs_since_last_improvement+=1
    
        epoch+=1

    return best_val_loss, epoch-1
