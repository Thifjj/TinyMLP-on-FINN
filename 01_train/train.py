import torch
import torch.nn as nn
import torch.optim as optim

from model import TinyMLP


torch.manual_seed(42)

# =========================
# Dataset sintético
# =========================

NUM_SAMPLES = 5000

X = torch.randint(
    low=0,
    high=2,
    size=(NUM_SAMPLES, 16)
).float()

# classe 1 se houver mais de 8 bits = 1
# classe 0 caso contrário
y = (X.sum(dim=1) > 8).long()


# =========================
# Train / Test
# =========================

split = int(NUM_SAMPLES * 0.8)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]


# =========================
# Modelo
# =========================

model = TinyMLP()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)


# =========================
# Treinamento
# =========================

EPOCHS = 100

for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()

    output = model(X_train)

    loss = criterion(output, y_train)

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:

        prediction = output.argmax(dim=1)

        accuracy = (
            prediction == y_train
        ).float().mean()

        print(
            f"Epoch {epoch+1:3d} | "
            f"Loss: {loss.item():.4f} | "
            f"Acc: {accuracy.item()*100:.2f}%"
        )


# =========================
# Teste
# =========================

model.eval()

with torch.no_grad():

    output = model(X_test)

    prediction = output.argmax(dim=1)

    accuracy = (
        prediction == y_test
    ).float().mean()

print()
print("=========================")
print("Resultado")
print("=========================")

print(
    f"Accuracy teste: "
    f"{accuracy.item()*100:.2f}%"
)


# =========================
# Salvar pesos
# =========================

torch.save(
    model.state_dict(),
    "tiny_mlp.pth"
)

print()
print("Modelo salvo em:")
print("tiny_mlp.pth")