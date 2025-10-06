import os
# ✅ Fix OMP duplicate error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
from torchvision import models, transforms
from PIL import Image, ImageTk

# Fixed image folder path
IMAGE_FOLDER = r"C:\Users\Hp\Desktop\q7_solution\images"


class CatDogClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat vs Dog Classifier")
        self.root.geometry("800x600")

        self.model = self.load_model()
        self.preprocess = self.get_preprocess()
        self.image_paths = []

        self.setup_gui()

    def load_model(self):
        """Load pretrained ResNet50"""
        try:
            model = models.resnet50(pretrained=True)
            model.eval()
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return None

    def get_preprocess(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def setup_gui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Cat vs Dog Classifier", font=("Arial", 16, "bold")).pack(pady=10)

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Load from Folder", command=self.load_from_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Analyze", command=self.analyze_images).pack(side=tk.LEFT, padx=5)

        # Results table
        columns = ("Image", "Prediction", "Confidence")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=12)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=180)
        self.tree.pack(fill="both", expand=True, pady=10)

        self.image_label = ttk.Label(frame, text="Image preview will appear here")
        self.image_label.pack(pady=10)

        self.tree.bind("<<TreeviewSelect>>", self.show_selected_image)

    def load_from_folder(self):
        """Load images from fixed folder path"""
        if not os.path.exists(IMAGE_FOLDER):
            messagebox.showerror("Error", f"Folder not found:\n{IMAGE_FOLDER}")
            return

        all_files = os.listdir(IMAGE_FOLDER)
        image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            messagebox.showwarning("No Images", "No .jpg/.png images found in the folder.")
            return

        self.image_paths = [os.path.join(IMAGE_FOLDER, f) for f in image_files]
        messagebox.showinfo("Images Loaded", f"Loaded {len(self.image_paths)} image(s).")

    def clear_all(self):
        """Clear the table + preview, but keep loaded images"""
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.image_label.config(image="", text="Image preview will appear here")

    def classify(self, path):
        try:
            image = Image.open(path).convert("RGB")
            input_tensor = self.preprocess(image).unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probs, 1)

            class_idx = top_catid[0].item()
            # ImageNet classes 281–285 are cats
            class_name = "Cat" if 281 <= class_idx <= 285 else "Dog"
            return class_name, top_prob.item(), image
        except Exception as e:
            print(f"⚠️ Error classifying {path}: {e}")
            return "Error", 0, None

    def analyze_images(self):
        if not self.image_paths:
            messagebox.showwarning("No Images", "Please load images first.")
            return

        # Only clear table, not image list
        for i in self.tree.get_children():
            self.tree.delete(i)

        processed = 0
        for path in self.image_paths:
            name = os.path.basename(path)
            pred, conf, _ = self.classify(path)
            if pred != "Error":
                processed += 1
                self.tree.insert("", tk.END, values=(name, pred, f"{conf:.1%}"))

        messagebox.showinfo("Done", f"Processed {processed} out of {len(self.image_paths)} image(s).")

    def show_selected_image(self, event):
        item = self.tree.selection()
        if not item:
            return
        filename = self.tree.item(item[0], "values")[0]
        for path in self.image_paths:
            if os.path.basename(path) == filename:
                img = Image.open(path)
                img.thumbnail((300, 300))
                photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo  # keep reference
                break


def main():
    root = tk.Tk()
    CatDogClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
