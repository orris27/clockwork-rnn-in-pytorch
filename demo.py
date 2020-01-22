import torch
from crnn import CRNN

embedding_size = 8
hidden_state = 512
hidden_state = 4
output_size = 8
output_size = 2
batch_size = 1


clock_periods = list([2 ** i for i in range(9)])
model = CRNN(embedding_size, hidden_state, output_size, clock_periods)
inputs = torch.randn(16, batch_size, embedding_size)
y_predicted = model.forward(inputs)
print(y_predicted)

