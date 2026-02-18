# smart_patient_app.py
import json
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import csv
import cv2

DATA_FILE = "patients.json"
MODEL_FILE = "model.pkl"
SYNTH_CSV = "health_data.csv"

# -------------------- storage -------------------- #
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")

# -------------------- ML utilities -------------------- #
def generate_synthetic_health_data(path=SYNTH_CSV, n=1000, seed=42):
    """
    Generate a synthetic health dataset with features:
    age, bmi, systolic_bp, diastolic_bp, glucose, smoker(0/1), diabetes(0/1)
    target: risk (0:Low,1:Medium,2:High)
    """
    import random
    random.seed(seed)
    header = ["age","bmi","sys_bp","dia_bp","glucose","smoker","diabetes","risk"]
    rows = []
    for _ in range(n):
        age = random.randint(18, 90)
        bmi = round(random.uniform(16.0, 40.0),1)
        sys_bp = random.randint(90, 190)
        dia_bp = max(50, sys_bp - random.randint(30,70))
        glucose = random.randint(70, 300)
        smoker = random.choices([0,1], weights=[0.7,0.3])[0]
        diabetes = 1 if glucose > 140 or random.random() < 0.05 else 0

        # simple rule-based risk to label synthetic output
        score = 0
        if age > 60: score += 1
        if bmi > 30: score += 1
        if sys_bp > 140: score += 1
        if glucose > 180: score += 2
        if smoker: score += 1
        if diabetes: score += 2

        if score <= 1:
            risk = 0
        elif score <= 3:
            risk = 1
        else:
            risk = 2
        rows.append([age,bmi,sys_bp,dia_bp,glucose,smoker,diabetes,risk])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path

def load_dataset(path=SYNTH_CSV):
    X = []
    y = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            X.append([
                float(r["age"]), float(r["bmi"]), float(r["sys_bp"]),
                float(r["dia_bp"]), float(r["glucose"]), float(r["smoker"]),
                float(r["diabetes"])
            ])
            y.append(int(r["risk"]))
    return np.array(X), np.array(y)

def train_or_load_model(force_retrain=False):
    # If model exists and not forcing retrain, load it
    if os.path.exists(MODEL_FILE) and not force_retrain:
        try:
            model = joblib.load(MODEL_FILE)
            return model, None
        except Exception:
            pass

    # If user supplied health_data.csv, use it; otherwise generate synthetic
    if not os.path.exists(SYNTH_CSV):
        generate_synthetic_health_data(SYNTH_CSV, n=1200)

    X, y = load_dataset(SYNTH_CSV)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    joblib.dump(model, MODEL_FILE)
    return model, acc

def predict_risk_from_features(model, features):
    """
    features: [age, bmi, sys_bp, dia_bp, glucose, smoker, diabetes]
    returns (label, probability_array)
    """
    arr = np.array(features, dtype=float).reshape(1,-1)
    probs = model.predict_proba(arr)[0]
    pred = int(model.predict(arr)[0])
    labels = {0:"Low", 1:"Medium", 2:"High"}
    return labels.get(pred,"Unknown"), probs

# -------------------- Image processing -------------------- #
def process_medical_image(path):
    """
    Returns a dict of processed images (PIL Image objects) for display:
    original, gray, blurred, edges, enhanced
    """
    img_cv = cv2.imread(path)
    if img_cv is None:
        raise ValueError("Unable to read image")
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # Edge detection
    edges = cv2.Canny(blur, 50, 150)
    # Contrast stretch / CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # Convert to PIL
    def cv2pil(img):
        if img.ndim == 2:
            return Image.fromarray(img)
        else:
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return {
        "original": cv2pil(img_cv),
        "gray": cv2pil(gray),
        "blur": cv2pil(blur),
        "edges": cv2pil(edges),
        "enhanced": cv2pil(enhanced)
    }

# -------------------- GUI App -------------------- #
class PatientApp:
    def __init__(self, root):
        self.root = root
        root.title("Patient Health Record System (Smart Edition)")
        root.geometry("1000x560")
        root.resizable(False, False)

        # Frames
        left = ttk.Frame(root, padding=(10,10))
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(root, padding=(10,10))
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Controls
        ttk.Label(left, text="Controls", font=("Helvetica", 12, "bold")).pack(pady=(0,8))
        ttk.Button(left, text="Add Patient", width=22, command=self.open_add_dialog).pack(pady=4)
        ttk.Button(left, text="Edit Selected", width=22, command=self.edit_selected).pack(pady=4)
        ttk.Button(left, text="Delete Selected", width=22, command=self.delete_selected).pack(pady=4)
        ttk.Button(left, text="Predict Risk (AI)", width=22, command=self.open_predict_dialog).pack(pady=6)
        ttk.Button(left, text="Upload Image (X-ray demo)", width=22, command=self.open_image_dialog).pack(pady=6)
        ttk.Button(left, text="Export JSON...", width=22, command=self.export_json).pack(pady=4)
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(left, text="Search", font=("Helvetica", 11)).pack(pady=(6,4))
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(left, textvariable=self.search_var, width=24)
        search_entry.pack()
        search_entry.bind("<Return>", lambda e: self.refresh_list())
        ttk.Button(left, text="Search", width=22, command=self.refresh_list).pack(pady=6)
        ttk.Button(left, text="Show All", width=22, command=lambda: self.refresh_list(show_all=True)).pack(pady=2)

        # Right: Tree + details
        ttk.Label(right, text="Patients", font=("Helvetica", 12, "bold")).pack(anchor=tk.W)
        columns = ("id","name","age","gender","disease","contact","added_on")
        self.tree = ttk.Treeview(right, columns=columns, show="headings", height=18)
        for col, text, w in [("id","ID",40),("name","Name",180),("age","Age",60),("gender","Gender",80),("disease","Disease/Condition",220),("contact","Contact",120),("added_on","Added On",160)]:
            self.tree.heading(col, text=text)
            self.tree.column(col, width=w, anchor=tk.CENTER if col in ("id","age") else tk.W)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        ttk.Label(right, text="Details", font=("Helvetica", 12, "bold")).pack(anchor=tk.W, pady=(8,0))
        self.details_text = tk.Text(right, height=7, wrap=tk.WORD)
        self.details_text.pack(fill=tk.X)
        self.details_text.configure(state="disabled")

        # Load data & model
        self.data = load_data()
        self.model, self.training_info = train_or_load_model()
        if self.training_info is not None:
            print(f"Model trained and saved to {MODEL_FILE} (accuracy on test set: {self.training_info:.3f})")
        else:
            print(f"Loaded existing model from {MODEL_FILE}")

        self.refresh_list(show_all=True)

    def refresh_list(self, show_all=False):
        q = self.search_var.get().strip().lower()
        for i in self.tree.get_children():
            self.tree.delete(i)
        for idx, p in enumerate(self.data, start=1):
            name = p.get("Name","")
            if show_all or q == "" or q in name.lower() or q in p.get("Disease","").lower():
                self.tree.insert("", tk.END, iid=str(idx-1), values=(
                    idx,
                    name,
                    p.get("Age",""),
                    p.get("Gender",""),
                    p.get("Disease",""),
                    p.get("Contact",""),
                    p.get("Created_At","")
                ))

    def on_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        p = self.data[idx]
        detail = (
            f"Name: {p.get('Name','')}\n"
            f"Age: {p.get('Age','')}\n"
            f"Gender: {p.get('Gender','')}\n"
            f"Contact: {p.get('Contact','')}\n"
            f"Disease/Condition: {p.get('Disease','')}\n"
            f"Prescription: {p.get('Prescription','')}\n"
            f"Notes: {p.get('Notes','')}\n"
            f"Added On: {p.get('Created_At','')}\n"
            f"Last Updated: {p.get('Updated_At','') if p.get('Updated_At') else 'N/A'}\n"
        )
        self.details_text.configure(state="normal")
        self.details_text.delete("1.0", tk.END)
        self.details_text.insert(tk.END, detail)
        self.details_text.configure(state="disabled")

    def open_add_dialog(self):
        dialog = PatientDialog(self.root, "Add New Patient")
        if dialog.result:
            p = dialog.result
            p["Created_At"] = timestamp()
            p["Updated_At"] = ""
            self.data.append(p)
            save_data(self.data)
            self.refresh_list(show_all=True)
            messagebox.showinfo("Success", f"Record for '{p['Name']}' added.")

    def edit_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("No selection", "Please select a patient to edit.")
            return
        idx = int(sel[0])
        p = self.data[idx]
        dialog = PatientDialog(self.root, "Edit Patient", p)
        if dialog.result:
            newp = dialog.result
            newp["Created_At"] = p.get("Created_At", timestamp())
            newp["Updated_At"] = timestamp()
            self.data[idx] = newp
            save_data(self.data)
            self.refresh_list(show_all=True)
            messagebox.showinfo("Updated", f"Record for '{newp['Name']}' updated.")

    def delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("No selection", "Please select a patient to delete.")
            return
        idx = int(sel[0])
        p = self.data[idx]
        confirm = messagebox.askyesno("Confirm Delete", f"Delete record for '{p.get('Name')}'?")
        if confirm:
            del self.data[idx]
            save_data(self.data)
            self.refresh_list(show_all=True)
            self.details_text.configure(state="normal")
            self.details_text.delete("1.0", tk.END)
            self.details_text.configure(state="disabled")
            messagebox.showinfo("Deleted", "Record deleted successfully.")

    def export_json(self):
        path = filedialog.asksaveasfilename(defaultextension=".json",
            filetypes=[("JSON files","*.json"),("All files","*.*")], title="Export JSON")
        if path:
            try:
                with open(path, "w") as f:
                    json.dump(self.data, f, indent=4)
                messagebox.showinfo("Exported", f"Data exported to {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")

    # --------- NEW: Predict dialog --------- #
    def open_predict_dialog(self):
        dlg = PredictDialog(self.root, self.model)
        if dlg.result:
            label, probs = dlg.result
            messagebox.showinfo("Prediction", f"Risk level: {label}\nProbabilities (Low/Med/High): {probs.round(3)}")

    # --------- NEW: Image processing dialog --------- #
    def open_image_dialog(self):
        path = filedialog.askopenfilename(title="Select medical image (X-ray/MRI)", filetypes=[("Image files","*.png;*.jpg;*.jpeg;*.bmp"),("All files","*.*")])
        if not path:
            return
        try:
            imgs = process_medical_image(path)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot process image: {e}")
            return

        # Create display window
        win = tk.Toplevel(self.root)
        win.title("Image Processing Results")
        # show thumbnails
        canvas = tk.Canvas(win, width=950, height=600)
        canvas.pack()
        # prepare images
        thumbs = []
        x = 10
        y = 10
        for title in ("original","enhanced","gray","blur","edges"):
            pil = imgs[title].copy()
            pil.thumbnail((300,300))
            tkimg = ImageTk.PhotoImage(pil)
            thumbs.append(tkimg)  # keep reference
            canvas.create_image(x, y, anchor=tk.NW, image=tkimg)
            canvas.create_text(x+150, y+310, text=title.capitalize(), font=("Helvetica",10,"bold"))
            x += 310
            if x + 300 > 950:
                x = 10
                y += 350
        # keep reference on window
        win._thumbs = thumbs

# --------- Predict dialog UI --------- #
class PredictDialog(simpledialog.Dialog):
    def __init__(self, parent, model):
        self.model = model
        super().__init__(parent, "Predict Disease Risk from Vitals")

    def body(self, master):
        ttk.Label(master, text="Age:").grid(row=0, column=0, sticky=tk.W, pady=4)
        ttk.Label(master, text="BMI:").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Label(master, text="Systolic BP:").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Label(master, text="Diastolic BP:").grid(row=3, column=0, sticky=tk.W, pady=4)
        ttk.Label(master, text="Glucose:").grid(row=4, column=0, sticky=tk.W, pady=4)
        ttk.Label(master, text="Smoker (0/1):").grid(row=5, column=0, sticky=tk.W, pady=4)
        ttk.Label(master, text="Diabetes (0/1):").grid(row=6, column=0, sticky=tk.W, pady=4)

        self.e_age = ttk.Entry(master, width=20)
        self.e_bmi = ttk.Entry(master, width=20)
        self.e_sys = ttk.Entry(master, width=20)
        self.e_dia = ttk.Entry(master, width=20)
        self.e_glu = ttk.Entry(master, width=20)
        self.e_sm = ttk.Entry(master, width=20)
        self.e_diab = ttk.Entry(master, width=20)

        self.e_age.grid(row=0, column=1, pady=4)
        self.e_bmi.grid(row=1, column=1, pady=4)
        self.e_sys.grid(row=2, column=1, pady=4)
        self.e_dia.grid(row=3, column=1, pady=4)
        self.e_glu.grid(row=4, column=1, pady=4)
        self.e_sm.grid(row=5, column=1, pady=4)
        self.e_diab.grid(row=6, column=1, pady=4)

        # helpful defaults
        self.e_age.insert(0, "45")
        self.e_bmi.insert(0, "24.5")
        self.e_sys.insert(0, "120")
        self.e_dia.insert(0, "80")
        self.e_glu.insert(0, "100")
        self.e_sm.insert(0, "0")
        self.e_diab.insert(0, "0")

        return self.e_age

    def validate(self):
        try:
            # basic parse to ensure numeric
            _ = float(self.e_age.get())
            _ = float(self.e_bmi.get())
            _ = float(self.e_sys.get())
            _ = float(self.e_dia.get())
            _ = float(self.e_glu.get())
            _ = int(self.e_sm.get())
            _ = int(self.e_diab.get())
            return True
        except Exception:
            messagebox.showwarning("Validation", "Please enter valid numeric values.")
            return False

    def apply(self):
        features = [
            float(self.e_age.get()),
            float(self.e_bmi.get()),
            float(self.e_sys.get()),
            float(self.e_dia.get()),
            float(self.e_glu.get()),
            int(self.e_sm.get()),
            int(self.e_diab.get())
        ]
        label, probs = predict_risk_from_features(self.model, features)
        self.result = (label, probs)

# --------- Patient dialog (same as before) --------- #
class PatientDialog(simpledialog.Dialog):
    def __init__(self, parent, title, data=None):
        self.data = data
        super().__init__(parent, title)

    def body(self, master):
        ttk.Label(master, text="Name:").grid(row=0, column=0, sticky=tk.W, pady=4)
        ttk.Label(master, text="Age:").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Label(master, text="Gender:").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Label(master, text="Contact:").grid(row=3, column=0, sticky=tk.W, pady=4)
        ttk.Label(master, text="Disease/Condition:").grid(row=4, column=0, sticky=tk.W, pady=4)
        ttk.Label(master, text="Prescription:").grid(row=5, column=0, sticky=tk.W, pady=4)
        ttk.Label(master, text="Notes:").grid(row=6, column=0, sticky=tk.W, pady=4)

        self.e_name = ttk.Entry(master, width=40)
        self.e_age = ttk.Entry(master, width=20)
        self.e_gender = ttk.Entry(master, width=20)
        self.e_contact = ttk.Entry(master, width=30)
        self.e_disease = ttk.Entry(master, width=40)
        self.e_presc = ttk.Entry(master, width=50)
        self.e_notes = ttk.Entry(master, width=60)

        self.e_name.grid(row=0, column=1, pady=4, columnspan=2)
        self.e_age.grid(row=1, column=1, pady=4, sticky=tk.W)
        self.e_gender.grid(row=2, column=1, pady=4, sticky=tk.W)
        self.e_contact.grid(row=3, column=1, pady=4, sticky=tk.W)
        self.e_disease.grid(row=4, column=1, pady=4, columnspan=2, sticky=tk.W)
        self.e_presc.grid(row=5, column=1, pady=4, columnspan=2, sticky=tk.W)
        self.e_notes.grid(row=6, column=1, pady=4, columnspan=2, sticky=tk.W)

        if self.data:
            self.e_name.insert(0, self.data.get("Name",""))
            self.e_age.insert(0, self.data.get("Age",""))
            self.e_gender.insert(0, self.data.get("Gender",""))
            self.e_contact.insert(0, self.data.get("Contact",""))
            self.e_disease.insert(0, self.data.get("Disease",""))
            self.e_presc.insert(0, self.data.get("Prescription",""))
            self.e_notes.insert(0, self.data.get("Notes",""))

        return self.e_name

    def validate(self):
        name = self.e_name.get().strip()
        if name == "":
            messagebox.showwarning("Validation", "Name cannot be empty.")
            return False
        return True

    def apply(self):
        self.result = {
            "Name": self.e_name.get().strip(),
            "Age": self.e_age.get().strip(),
            "Gender": self.e_gender.get().strip(),
            "Contact": self.e_contact.get().strip(),
            "Disease": self.e_disease.get().strip(),
            "Prescription": self.e_presc.get().strip(),
            "Notes": self.e_notes.get().strip()
        }

if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        save_data([])

    root = tk.Tk()
    app = PatientApp(root)
    root.mainloop()
