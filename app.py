import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from predict import predict_image

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Tiếng Việt")
        self.root.geometry("400x500")  # Tăng chiều cao để căn chỉnh tốt hơn
        
        # Tạo frame chính để căn giữa
        main_frame = tk.Frame(root, padx=20, pady=20)
        main_frame.pack(expand=True)

        # Tiêu đề
        title_label = tk.Label(main_frame, text="Tải ảnh để nhận diện văn bản", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        # Frame cho ảnh
        image_frame = tk.Frame(main_frame)
        image_frame.pack(pady=20)
        
        self.image_label = tk.Label(image_frame)
        self.image_label.pack()

        # Frame cho kết quả
        result_frame = tk.Frame(main_frame)
        result_frame.pack(pady=20)
        
        self.result_label = tk.Label(result_frame, text="Kết quả: ", font=("Arial", 12))
        self.result_label.pack()

        # Frame cho nút
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        self.upload_button = tk.Button(button_frame, text="Tải ảnh", command=self.upload_image, font=("Arial", 10), padx=10, pady=5)
        self.upload_button.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            # Hiển thị ảnh
            img = Image.open(file_path)
            img = img.resize((200, 100))
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img)
            self.image_label.image = img
            
            # Dự đoán
            try:
                text = predict_image(file_path)
                self.result_label.config(text=f"Kết quả: {text}")
            except Exception as e:
                self.result_label.config(text=f"Kết quả: Lỗi - {str(e)}")
                messagebox.showerror("Lỗi", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()