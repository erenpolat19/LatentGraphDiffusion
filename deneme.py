from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.data import DataLoader

# Load the molhiv dataset
dataset = PygGraphPropPredDataset(name='ogbg-molhiv')

# Access the graph data and the task (i.e., the property we want to predict)
data = dataset[0]  # Access the first graph in the dataset


# Extract the input features and target

print(data)
print(data.x)