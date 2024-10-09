import os
import cv2
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tkinterdnd2 import TkinterDnD, DND_FILES
from ultralytics import YOLO
import traceback


class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection Image Viewer")
        self.root.configure(bg='#d3d3d3')

        self.model = None
        self.image_list = []
        self.image_index = 0
        self.img_tk = None
        self.img_pil = None
        self.placeholder_text_id = None
        self.link_text_id = None

        # Tworzenie layoutu aplikacji
        self.create_widgets()

        # Bindowanie zmiany rozmiaru canvas
        self.canvas.bind("<Configure>", self.on_resize)

        # Wyświetlanie komunikatu i funkcjonalności wyboru folderu
        self.show_placeholder_text()

        # Obsługa przeciągania i upuszczania folderu
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.drop_folder)

        # Dodanie eleganckiego zamknięcia aplikacji
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """Tworzy elementy GUI."""
        self.top_frame = Frame(self.root, bg='#d3d3d3')
        self.top_frame.pack(side=TOP, fill=BOTH, expand=True)

        self.bottom_frame = Frame(self.root, bg='#d3d3d3')
        self.bottom_frame.pack(side=BOTTOM, fill=X, pady=10)

        # Canvas do wyświetlania obrazów
        self.canvas = Canvas(self.top_frame, bg='#444444', highlightthickness=1,
                             highlightbackground='#cccccc', width=640, height=640)
        self.canvas.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Przyciski
        btn_style_prev_next = self.get_button_style('#777777', '#999999', 6)
        btn_style_open = self.get_button_style('#5A9BD3', '#4A82C2', 16)

        # Ramka dla przycisków nawigacyjnych
        self.left_frame = Frame(self.bottom_frame, bg='#d3d3d3')
        self.left_frame.pack(side=LEFT, padx=10)

        self.nav_frame = Frame(self.left_frame, bg='#d3d3d3')
        self.nav_frame.pack(side=TOP, pady=5)

        self.btn_prev = Button(self.nav_frame, text="⟵", command=self.show_prev_image, **btn_style_prev_next)
        self.btn_prev.pack(side=LEFT, padx=15)

        self.btn_next = Button(self.nav_frame, text="⟶", command=self.show_next_image, **btn_style_prev_next)
        self.btn_next.pack(side=LEFT, padx=15)

        self.btn_open = Button(self.left_frame, text="Open Folder", command=self.open_folder, **btn_style_open)
        self.btn_open.pack(side=TOP, pady=(10, 0))

        # Log tekstowy do wyświetlania wykrytych obiektów
        self.log_frame = Frame(self.bottom_frame, bg='#d3d3d3')
        self.log_frame.pack(side=RIGHT, padx=10, fill=BOTH, expand=True)

        self.text_log = Text(self.log_frame, height=5, width=50, font=('Arial', 10), wrap=WORD, bg='#AAAAAA',
                             fg='black', relief='flat', bd=0)
        self.text_log.pack(side=LEFT, padx=60)

        self.scrollbar = Scrollbar(self.log_frame, command=self.text_log.yview, width=0)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.text_log.config(yscrollcommand=self.scrollbar.set)

    def get_button_style(self, bg_color, active_bg_color, width):
        """Funkcja pomocnicza do uzyskania stylu przycisków."""
        return {
            'bg': bg_color,
            'fg': 'white',
            'font': ('Arial', 12, 'bold'),
            'relief': 'flat',
            'bd': 2,
            'activebackground': active_bg_color,
            'activeforeground': 'white',
            'width': width
        }

    def load_model(self):
        """Ładuje model YOLO."""
        try:
            yolo_model_path = os.path.join(os.curdir, 'model', 'best.pt')
            self.model = YOLO(yolo_model_path)
        except Exception as e:
            self.text_log.insert(END, f"Error loading model: {str(e)}\n")
            traceback.print_exc()

    def open_folder(self):
        """Otwiera folder i ładuje obrazy do przetwarzania."""
        self.load_model()
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                               if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            self.image_index = 0
            self.show_image()
        else:
            self.show_placeholder_text()

    def show_placeholder_text(self):
        """Wyświetla tekst zachęcający do wybrania folderu."""
        self.canvas.delete("all")
        self.placeholder_text_id = self.canvas.create_text(320, 320, text="or drop directory here", fill="white",
                                                           font=('Arial', 16), anchor=CENTER)
        self.link_text_id = self.canvas.create_text(320, 290, text="Choose a directory", fill="#5a9bd3",
                                                    font=('Arial', 16, 'underline'), anchor=CENTER)

        self.canvas.tag_bind(self.link_text_id, "<Enter>", lambda e: self.canvas.config(cursor="hand2"))
        self.canvas.tag_bind(self.link_text_id, "<Leave>", lambda e: self.canvas.config(cursor=""))
        self.canvas.tag_bind(self.link_text_id, "<Button-1>", lambda event: self.open_folder())

    def drop_folder(self, event):
        """Obsługuje przeciąganie i upuszczanie folderu."""
        try:
            self.load_model()
            folder_path = event.data.strip('{}')
            if os.path.isdir(folder_path):
                self.image_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                                   if f.lower().endswith(('png', 'jpg', 'jpeg'))]
                self.image_index = 0
                self.show_image()

        except Exception as e:
            self.text_log.insert(END, f"Error loading folder or displaying img: {str(e)}\n")
            traceback.print_exc()

    def show_image(self):
        """Wyświetla bieżący obraz z wykrytymi obiektami."""
        if not self.image_list:
            return
        try:
            self.clear_placeholder()
            if self.model is None:
                raise ValueError("YOLO model not loaded. Please check the model path.")
            image_path = self.image_list[self.image_index]
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load img: {image_path}")
            annotated_img = self.detect_objects(img)
            self.display_image(annotated_img)
        except Exception as e:
            self.text_log.insert(END, f"Error displaying img: {str(e)}\n")
            traceback.print_exc()

    def clear_placeholder(self):
        """Usuwa placeholder, jeśli istnieje."""
        if self.placeholder_text_id is not None:
            self.canvas.delete(self.placeholder_text_id)
            self.canvas.delete(self.link_text_id)
            self.placeholder_text_id = None
            self.link_text_id = None

    def detect_objects(self, img):
        """Wykrywa obiekty w obrazie i zwraca obraz z adnotacjami."""
        results = self.model(img)
        self.update_log(results[0].boxes.data.cpu().numpy(), self.model.names)
        return results[0].plot()

    def display_image(self, annotated_img):
        """Wyświetla obraz na canvas."""
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        self.img_pil = Image.fromarray(annotated_img)
        self.update_canvas()

    def update_canvas(self):
        """Aktualizuje obraz na canvas po przeskalowaniu."""
        if self.img_pil:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_width, img_height = self.img_pil.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            resized_img = self.img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

            self.img_tk = ImageTk.PhotoImage(image=resized_img)
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=CENTER, image=self.img_tk)

    def show_prev_image(self):
        """Pokazuje poprzedni obraz w folderze."""
        if self.image_list:
            self.image_index = (self.image_index - 1) % len(self.image_list)
            self.show_image()

    def show_next_image(self):
        """Pokazuje następny obraz w folderze."""
        if self.image_list:
            self.image_index = (self.image_index + 1) % len(self.image_list)
            self.show_image()

    def update_log(self, detections, class_names):
        """Aktualizuje log z informacją o wykrytych obiektach."""
        self.text_log.delete(1.0, END)
        self.text_log.tag_configure("large", font=('Arial', 12))

        label_counts = {}
        self.text_log.insert(END, "Detected:\n", "large")

        if class_names is not None:
            labels = [class_names[int(cls)] for cls in detections[:, 5]]
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1

        for label, count in label_counts.items():
            if count > 1:
                self.text_log.insert(END, f" ●  {count}x - {label}\n")
            else:
                self.text_log.insert(END, f" ●  {label}\n")

    def on_resize(self, event):
        """Obsługuje zmianę rozmiaru okna."""
        self.update_canvas()

    def on_closing(self):
        """Zamyka aplikację."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    root.geometry("700x800")
    app = YOLOApp(root)
    root.mainloop()
