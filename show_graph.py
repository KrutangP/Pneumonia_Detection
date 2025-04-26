import matplotlib.pyplot as plt
import pickle

HISTORY_PATH = "model/training_history.pkl"

def show_loss_graph():
    with open(HISTORY_PATH, 'rb') as f:
        history = pickle.load(f)

    # Plotting Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_accuracy_graph():
    with open(HISTORY_PATH, 'rb') as f:
        history = pickle.load(f)

    # Plotting Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    show_loss_graph()         # To show Loss graph
    show_accuracy_graph()     # To show Accuracy graph
