#%%
from src.trainer import Trainer

c = {
    'model_name': 'test',
    'seed': [0, 1],
    'bs': 32, 'lr': 1e-3, 'n_epoch': 2
}

trainer = Trainer(c)
trainer.fit()