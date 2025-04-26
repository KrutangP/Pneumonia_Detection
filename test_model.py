import torch
from torch.utils.data import DataLoader
from model_architecture import build_model
from data_loader import load_test_data  # You can use your modified function from data_loader.py
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set the device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model():
    test_loader = load_test_data(batch_size=32)

    model = build_model()
    model.load_state_dict(torch.load('model/pneumonia_model.pth'))
    model.to(device)
    model.eval()

    correct_preds = 0
    total_preds = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            predicted = (outputs.squeeze() > 0.5).float()

            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = correct_preds / total_preds * 100
    print(f"\n✅ Test Accuracy: {accuracy:.2f}%")

    class_names = ['NORMAL', 'PNEUMONIA']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    evaluate_model()
