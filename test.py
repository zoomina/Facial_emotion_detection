import torch

def test(self, X, y):
    X = torch.tensor(X, dtype=torch.float).to(self.device)
    y = torch.tensor(y, dtype=torch.float).to(self.device)

    self.model.eval()
    with torch.no_grad():
        test_summary = {"loss": [], "output": []}
        for i in range(X.shape[0]):
            output = self.model(X[i].unsqueeze(dim=0))
            loss = self.loss(output, y[i])
            test_summary["loss"].append(loss.detach().cpu().numpy())

        print("Test complete : avg. loss :", sum(test_summary["loss"]) / len(test_summary["loss"]))
