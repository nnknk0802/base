#%%
from sklearn.model_selection import train_test_split

import torch.utils.data as data
import torchvision
from torchvision import transforms
from PIL import Image

class Dataset(data.Dataset):
    def __init__(self):
        super().__init__()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        with open(path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        return x

def load_dataloaders(batch_size=64, seed=0):
    dataset = data.TensorDataset(torch.tensor(x), torch.tensor(y))
    idx = range(10)
    datasets, dataloaders, indices = {}, {}, {}
    indices['train'], indices['val'] = train_test_split(idx, test_size=0.2, random_state=seed)
    for phase in ['train', 'val']:
        datasets[phase] = data.Subset(dataset, indices[phase])
        dataloaders[phase] = data.DataLoader(datasets[phase], batch_size=batch_size, shuffle=True)
    return datasets, dataloaders