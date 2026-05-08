import torch

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

def test_attack(model, data_loader, epsilon):
    correct = 0
    total = 0

    for data, target in data_loader:
        data.requires_grad = True
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)

        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        output = model(perturbed_data)
        final_pred = output.argmax(dim=1)

        correct += (final_pred == target).sum().item()
        total += target.size(0)

    print(f"Accuracy under attack: {correct/total}")
