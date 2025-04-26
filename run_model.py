import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from torchvision import transforms
from model_architecture import build_model
from PIL import Image
import os

MODEL_PATH = 'project/model/pneumonia_model.pth'

def predict_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    img = Image.open(img_path).convert('L')
    img = transform(img).unsqueeze(0)

    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with torch.no_grad():
        output = model(img)
        prediction = output.item() > 0.5

    return prediction

# Custom Output Window
def show_result_window(result):
    output_window = tk.Toplevel()  # Create a new top-level window
    output_window.title("Prediction Result")
    output_window.geometry("400x300")
    output_window.configure(bg="#D1F2F6")  # Light blue background
    
    message = "Pneumonia Detected" if result else "Normal"
    
    # Display message with styled font
    result_label = tk.Label(output_window, text=message, font=("Helvetica", 20, "bold"), bg="#D1F2F6", fg="#E74C3C" if result else "#2ECC71")
    result_label.pack(pady=20)

    # Ask the user if they want to see the training graph
    def show_graph():
        os.system("python show_graph.py")  # Open the graph script

    # Display the prompt asking the user
    def on_show_graph_response(response):
        if response == 'yes':
            show_graph()
        output_window.destroy()  # Close the result window after user response

    # Add label for graph prompt
    graph_prompt_label = tk.Label(output_window, text="Would you like to see the training graph?", font=("Helvetica", 12), bg="#D1F2F6")
    graph_prompt_label.pack(pady=10)

    # Add a minimalistic frame for the Yes and No buttons
    button_frame = tk.Frame(output_window, bg="#D1F2F6")
    button_frame.pack(pady=10)

    # Add Yes and No buttons with minimal size
    yes_button = tk.Button(button_frame, text="Yes", command=lambda: on_show_graph_response('yes'), font=("Helvetica", 10), bg="#4CAF50", fg="white", padx=10, pady=5)
    yes_button.pack(side="left", padx=10)

    no_button = tk.Button(button_frame, text="No", command=lambda: on_show_graph_response('no'), font=("Helvetica", 10), bg="#E74C3C", fg="white", padx=10, pady=5)
    no_button.pack(side="left", padx=10)

    # Button to close the output window, placed at the bottom
    close_button = tk.Button(output_window, text="Close", command=output_window.destroy, font=("Helvetica", 10), bg="#34495E", fg="white", padx=10, pady=5)
    close_button.pack(pady=20)

    # Show the output window
    output_window.mainloop()

def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        result = predict_image(file_path)
        show_result_window(result)

# UI
root = tk.Tk()
root.title("Pneumonia Detector")
root.geometry("500x400")
root.configure(bg="#E8F0FE") 

# Title Label
title_label = tk.Label(root, text="Upload Chest X-ray Image", font=("Helvetica", 20, "bold"), bg="#E8F0FE", fg="#333")
title_label.place(relx=0.5, rely=0.3, anchor='center')

# Browse Button
browse_button = tk.Button(root, text="Browse", command=browse_file, 
                          font=("Helvetica", 14), bg="#4CAF50", fg="white", 
                          activebackground="#45a049", padx=20, pady=10, borderwidth=0)
browse_button.place(relx=0.5, rely=0.5, anchor='center')

root.mainloop()
