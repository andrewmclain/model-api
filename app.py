from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

# Define the PyTorch model (replace with your model)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

# Load the model
model = MyModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Define the API
app = FastAPI()

# Define input data structure
class InputData(BaseModel):
    data: list

@app.post("/predict/")
def predict(input_data: InputData):
    with torch.no_grad():
        tensor_data = torch.tensor(input_data.data)
        prediction = model(tensor_data).tolist()
        return {"prediction": prediction}

# Run the server using `uvicorn` command
