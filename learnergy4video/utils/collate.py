from torch.utils.data.dataloader import default_collate

def collate_fn(batches):
    # Try to remove audio from the batch
    try:
        batches = [(d[0], d[2]) for d in batches]
    except:
        batches = [(d[0], d[1]) for d in batches]
    return default_collate(batches)