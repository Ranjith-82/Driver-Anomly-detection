"""
Driver Management Application
A GUI app to manage driver profiles with photos and details

Features:
- Add driver photo
- Store driver information (ID, name, blood group, etc.)
- View driver profiles
- Save/Load driver database
- Search drivers
- Edit driver information
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import pickle
import os
from pathlib import Path
from datetime import datetime


class DriverApp:
    def __init__(self, root):
        """Initialize the Driver Management Application"""
        self.root = root
        self.root.title("Driver Management System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Data storage
        self.drivers_db = "drivers_database.pkl"
        self.photos_folder = "driver_photos"
        self.drivers = {}
        
        # Create required folders
        Path(self.photos_folder).mkdir(exist_ok=True)
        
        # Load existing drivers
        self.load_drivers()
        
        # Current selected driver
        self.current_driver_id = None
        self.current_photo = None
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Header
        header_frame = tk.Frame(self.root, bg="#1e3a8a", height=60)
        header_frame.pack(fill=tk.X)
        
        header_label = tk.Label(
            header_frame, 
            text="🚗 Driver Management System", 
            font=("Arial", 20, "bold"),
            bg="#1e3a8a",
            fg="white"
        )
        header_label.pack(pady=10)
        
        # Main content frame
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Driver list
        self.setup_left_panel(content_frame)
        
        # Right panel - Driver details
        self.setup_right_panel(content_frame)
    
    def setup_left_panel(self, parent):
        """Setup left panel with driver list"""
        left_frame = ttk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        # Title
        title_label = tk.Label(
            left_frame,
            text="Drivers List",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0"
        )
        title_label.pack(pady=5)
        
        # Search bar
        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.search_drivers)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20)
        search_entry.pack(side=tk.LEFT, padx=5)
        
        # Drivers listbox
        listbox_frame = ttk.Frame(left_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.drivers_listbox = tk.Listbox(
            listbox_frame,
            yscrollcommand=scrollbar.set,
            font=("Arial", 10),
            bg="white"
        )
        self.drivers_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.drivers_listbox.bind('<<ListboxSelect>>', self.on_driver_select)
        scrollbar.config(command=self.drivers_listbox.yview)
        
        # Buttons frame
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="+ New Driver", command=self.add_new_driver).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Delete Driver", command=self.delete_driver).pack(fill=tk.X, pady=2)
        
        self.refresh_drivers_list()
    
    def setup_right_panel(self, parent):
        """Setup right panel with driver details"""
        right_frame = ttk.Frame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Photo section
        photo_frame = ttk.LabelFrame(right_frame, text="Driver Photo", padding=10)
        photo_frame.pack(fill=tk.X, pady=5)
        
        self.photo_label = tk.Label(
            photo_frame,
            text="No Photo\n(Click to upload)",
            bg="#e0e0e0",
            width=20,
            height=15,
            font=("Arial", 10)
        )
        self.photo_label.pack(pady=10)
        self.photo_label.bind("<Button-1>", lambda e: self.upload_photo())
        
        photo_button_frame = ttk.Frame(photo_frame)
        photo_button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(photo_button_frame, text="Upload Photo", command=self.upload_photo).pack(side=tk.LEFT, padx=5)
        ttk.Button(photo_button_frame, text="Take Photo", command=self.take_photo).pack(side=tk.LEFT, padx=5)
        ttk.Button(photo_button_frame, text="Remove Photo", command=self.remove_photo).pack(side=tk.LEFT, padx=5)
        
        # Details section
        details_frame = ttk.LabelFrame(right_frame, text="Driver Information", padding=10)
        details_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create input fields
        self.details_fields = {}
        fields = [
            ("User ID", "user_id"),
            ("Full Name", "full_name"),
            ("Username", "username"),
            ("Email", "email"),
            ("Phone", "phone"),
            ("Blood Group", "blood_group"),
            ("License Number", "license_number"),
            ("Date of Birth", "dob"),
            ("Address", "address"),
            ("City", "city"),
            ("State", "state"),
            ("Pin Code", "pin_code"),
        ]
        
        row = 0
        for label_text, field_key in fields:
            ttk.Label(details_frame, text=label_text + ":").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
            
            if field_key == "address":
                # Multi-line text for address
                entry = tk.Text(details_frame, height=3, width=40, font=("Arial", 9))
                entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=5)
            else:
                entry = ttk.Entry(details_frame, width=40)
                entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=5)
            
            self.details_fields[field_key] = entry
            row += 1
        
        details_frame.columnconfigure(1, weight=1)
        
        # Buttons section
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Save Driver", command=self.save_driver).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Fields", command=self.clear_fields).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Profile", command=self.export_profile).pack(side=tk.LEFT, padx=5)
    
    def upload_photo(self):
        """Upload a photo from file"""
        file_path = filedialog.askopenfilename(
            title="Select Driver Photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_photo = file_path
            self.display_photo(file_path)
    
    def take_photo(self):
        """Capture photo using webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot access webcam!")
            return
        
        messagebox.showinfo("Webcam", "Press SPACE to capture photo, Press ESC to cancel")
        
        photo_path = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow("Take Driver Photo - Press SPACE to capture, ESC to cancel", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE key
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                photo_path = os.path.join(self.photos_folder, f"temp_{timestamp}.jpg")
                cv2.imwrite(photo_path, frame)
                messagebox.showinfo("Success", "Photo captured!")
                break
            elif key == 27:  # ESC key
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if photo_path and os.path.exists(photo_path):
            self.current_photo = photo_path
            self.display_photo(photo_path)
    
    def display_photo(self, photo_path):
        """Display photo in the label"""
        try:
            image = Image.open(photo_path)
            image.thumbnail((250, 350), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            self.photo_label.config(image=photo, text="")
            self.photo_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load photo: {e}")
    
    def remove_photo(self):
        """Remove the current photo"""
        self.current_photo = None
        self.photo_label.config(image="", text="No Photo\n(Click to upload)")
    
    def save_driver(self):
        """Save driver information"""
        user_id = self.details_fields["user_id"].get().strip()
        
        if not user_id:
            messagebox.showwarning("Warning", "Please enter User ID!")
            return
        
        # Collect all details
        driver_data = {
            "user_id": user_id,
            "full_name": self.details_fields["full_name"].get().strip(),
            "username": self.details_fields["username"].get().strip(),
            "email": self.details_fields["email"].get().strip(),
            "phone": self.details_fields["phone"].get().strip(),
            "blood_group": self.details_fields["blood_group"].get().strip(),
            "license_number": self.details_fields["license_number"].get().strip(),
            "dob": self.details_fields["dob"].get().strip(),
            "address": self.details_fields["address"].get(1.0, tk.END).strip(),
            "city": self.details_fields["city"].get().strip(),
            "state": self.details_fields["state"].get().strip(),
            "pin_code": self.details_fields["pin_code"].get().strip(),
            "photo": None,
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Handle photo
        if self.current_photo:
            # Copy photo to drivers folder
            filename = f"driver_{user_id}.jpg"
            photo_path = os.path.join(self.photos_folder, filename)
            
            img = Image.open(self.current_photo)
            # Convert to RGB if needed (JPEG doesn't support RGBA/alpha channels)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            img.save(photo_path)
            driver_data["photo"] = photo_path
        
        # Save to database
        self.drivers[user_id] = driver_data
        self.persist_drivers()
        
        messagebox.showinfo("Success", f"Driver '{user_id}' saved successfully!")
        self.refresh_drivers_list()
        self.clear_fields()
    
    def add_new_driver(self):
        """Add a new driver"""
        self.clear_fields()
        self.current_driver_id = None
        self.current_photo = None
        self.remove_photo()
        self.drivers_listbox.selection_clear(0, tk.END)
    
    def on_driver_select(self, event):
        """Handle driver selection from list"""
        selection = self.drivers_listbox.curselection()
        if not selection:
            return
        
        driver_id = self.drivers_listbox.get(selection[0]).split(" - ")[0]
        
        if driver_id in self.drivers:
            self.current_driver_id = driver_id
            self.load_driver_details(driver_id)
    
    def load_driver_details(self, driver_id):
        """Load driver details into form"""
        if driver_id not in self.drivers:
            return
        
        driver = self.drivers[driver_id]
        
        # Populate fields
        self.details_fields["user_id"].delete(0, tk.END)
        self.details_fields["user_id"].insert(0, driver.get("user_id", ""))
        
        self.details_fields["full_name"].delete(0, tk.END)
        self.details_fields["full_name"].insert(0, driver.get("full_name", ""))
        
        self.details_fields["username"].delete(0, tk.END)
        self.details_fields["username"].insert(0, driver.get("username", ""))
        
        self.details_fields["email"].delete(0, tk.END)
        self.details_fields["email"].insert(0, driver.get("email", ""))
        
        self.details_fields["phone"].delete(0, tk.END)
        self.details_fields["phone"].insert(0, driver.get("phone", ""))
        
        self.details_fields["blood_group"].delete(0, tk.END)
        self.details_fields["blood_group"].insert(0, driver.get("blood_group", ""))
        
        self.details_fields["license_number"].delete(0, tk.END)
        self.details_fields["license_number"].insert(0, driver.get("license_number", ""))
        
        self.details_fields["dob"].delete(0, tk.END)
        self.details_fields["dob"].insert(0, driver.get("dob", ""))
        
        self.details_fields["address"].delete(1.0, tk.END)
        self.details_fields["address"].insert(1.0, driver.get("address", ""))
        
        self.details_fields["city"].delete(0, tk.END)
        self.details_fields["city"].insert(0, driver.get("city", ""))
        
        self.details_fields["state"].delete(0, tk.END)
        self.details_fields["state"].insert(0, driver.get("state", ""))
        
        self.details_fields["pin_code"].delete(0, tk.END)
        self.details_fields["pin_code"].insert(0, driver.get("pin_code", ""))
        
        # Load photo
        if driver.get("photo") and os.path.exists(driver["photo"]):
            self.current_photo = driver["photo"]
            self.display_photo(driver["photo"])
        else:
            self.remove_photo()
    
    def delete_driver(self):
        """Delete selected driver"""
        selection = self.drivers_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a driver to delete!")
            return
        
        driver_id = self.drivers_listbox.get(selection[0]).split(" - ")[0]
        
        if messagebox.askyesno("Confirm", f"Delete driver '{driver_id}'?"):
            if driver_id in self.drivers:
                # Delete photo if exists
                if self.drivers[driver_id].get("photo"):
                    try:
                        os.remove(self.drivers[driver_id]["photo"])
                    except:
                        pass
                
                del self.drivers[driver_id]
                self.persist_drivers()
                messagebox.showinfo("Success", "Driver deleted!")
                self.refresh_drivers_list()
                self.clear_fields()
    
    def clear_fields(self):
        """Clear all input fields"""
        for field in self.details_fields.values():
            if isinstance(field, tk.Text):
                field.delete(1.0, tk.END)
            else:
                field.delete(0, tk.END)
        
        self.remove_photo()
        self.current_driver_id = None
        self.current_photo = None
    
    def search_drivers(self, *args):
        """Search drivers by name or ID"""
        search_term = self.search_var.get().lower()
        
        self.drivers_listbox.delete(0, tk.END)
        
        for driver_id, driver_info in self.drivers.items():
            if (search_term in driver_id.lower() or 
                search_term in driver_info.get("full_name", "").lower() or
                search_term in driver_info.get("username", "").lower()):
                display_text = f"{driver_id} - {driver_info.get('full_name', 'N/A')}"
                self.drivers_listbox.insert(tk.END, display_text)
    
    def refresh_drivers_list(self):
        """Refresh the drivers list"""
        search_term = self.search_var.get().lower()
        
        self.drivers_listbox.delete(0, tk.END)
        
        if not search_term:
            # Show all drivers
            for driver_id, driver_info in sorted(self.drivers.items()):
                display_text = f"{driver_id} - {driver_info.get('full_name', 'N/A')}"
                self.drivers_listbox.insert(tk.END, display_text)
        else:
            # Show filtered drivers
            self.search_drivers()
    
    def persist_drivers(self):
        """Save drivers database to file"""
        try:
            with open(self.drivers_db, 'wb') as f:
                pickle.dump(self.drivers, f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save database: {e}")
    
    def load_drivers(self):
        """Load drivers database from file"""
        try:
            if os.path.exists(self.drivers_db):
                with open(self.drivers_db, 'rb') as f:
                    self.drivers = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load database: {e}")
    
    def export_profile(self):
        """Export current driver profile as image"""
        if not self.current_driver_id:
            messagebox.showwarning("Warning", "Please select a driver first!")
            return
        
        driver_id = self.current_driver_id
        driver = self.drivers[driver_id]
        
        # Create a simple text profile
        profile_text = f"""
DRIVER PROFILE
{'='*50}

User ID: {driver.get('user_id', 'N/A')}
Name: {driver.get('full_name', 'N/A')}
Username: {driver.get('username', 'N/A')}
Email: {driver.get('email', 'N/A')}
Phone: {driver.get('phone', 'N/A')}
Blood Group: {driver.get('blood_group', 'N/A')}
License No: {driver.get('license_number', 'N/A')}
Date of Birth: {driver.get('dob', 'N/A')}
Address: {driver.get('address', 'N/A')}
City: {driver.get('city', 'N/A')}
State: {driver.get('state', 'N/A')}
Pin Code: {driver.get('pin_code', 'N/A')}
Created Date: {driver.get('created_date', 'N/A')}

{'='*50}
        """
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile=f"driver_profile_{driver_id}.txt"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(profile_text)
                messagebox.showinfo("Success", f"Profile exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")


def main():
    """Main function"""
    root = tk.Tk()
    app = DriverApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
