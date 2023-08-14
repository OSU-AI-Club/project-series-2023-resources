import torch
from torch import nn, save, load
from torch.optim import Adam
import torchvision
from emotion_model import EmotionModel
import data_pl

# IN PROGRESS

def train():
    model = EmotionModel().to('cpu')
    optimizer = Adam(model.parameters(), lr = 1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_set = data_pl.train_pl()

    for epoch in range(50): #train for 50 epochs
        for batch in train_set:
            X, y = batch
            X, y = X.to('cpu'), y.to('cpu')
            prediction = model(X)
            loss = loss_fn(prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}\n-------------------------------")
        print(f"\tloss:{loss}")
        print("--------------------------------------------")

    # saving our model to our environment

    return model

    # with open('model_state.pt', 'wb') as f:
    #     save(model.state_dict(), f)

