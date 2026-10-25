from __future__ import annotations

import json
import re
import shutil
import tkinter as tk
import uuid
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from driver_monitor.config import ProjectPaths
from driver_monitor.profiles.controller import DriverProfileController
from driver_monitor.profiles.model import DriverProfile
from driver_monitor.profiles.repository import DriverProfileRepository


PROFILE_FIELDS = (
    ("Full name", "full_name"),
    ("Username", "username"),
    ("Email", "email"),
    ("Phone", "phone"),
    ("Blood group", "blood_group"),
    ("License number", "license_number"),
    ("Date of birth", "date_of_birth"),
    ("City", "city"),
    ("State", "state"),
    ("PIN code", "pin_code"),
)


class DriverProfileApp:
    def __init__(
        self,
        root: tk.Tk,
        repository: DriverProfileRepository,
    ) -> None:
        self.root = root
        self.repository = repository
        self.controller = DriverProfileController(repository)
        self.current_user_id: str | None = None
        self.photo_source: Path | None = None
        self.photo_preview: object | None = None
        self.values = {key: tk.StringVar() for _, key in PROFILE_FIELDS}
        self.user_id = tk.StringVar()
        self.search_text = tk.StringVar()
        self._configure_window()
        self._build_ui()
        self._refresh_profiles()

    def _configure_window(self) -> None:
        self.root.title("Driver Profile Manager")
        self.root.geometry("1120x760")
        self.root.minsize(960, 640)
        self.root.configure(background="#f3f6fa")
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#f3f6fa")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("TLabel", background="#f3f6fa", foreground="#26364a")
        style.configure("Title.TLabel", font=("Segoe UI", 21, "bold"))
        style.configure("Subtitle.TLabel", foreground="#66758a")
        style.configure("Treeview", rowheight=31, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.configure("Primary.TButton", padding=(12, 8))

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, padding=(26, 20, 26, 12))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Driver profiles", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Manage driver details and photos stored on this device.",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, pady=(4, 0))

        content = ttk.Frame(self.root, padding=(20, 8, 20, 20))
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=1, minsize=330)
        content.columnconfigure(1, weight=2, minsize=550)
        content.rowconfigure(0, weight=1)

        list_card = ttk.Frame(content, style="Card.TFrame", padding=16)
        list_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ttk.Label(list_card, text="Driver directory", font=("Segoe UI", 13, "bold"),
                  background="#ffffff").pack(anchor=tk.W, pady=(0, 10))
        search = ttk.Entry(list_card, textvariable=self.search_text)
        search.pack(fill=tk.X, pady=(0, 10))
        self.search_text.trace_add("write", lambda *_: self._refresh_profiles())
        table = ttk.Frame(list_card, style="Card.TFrame")
        table.pack(fill=tk.BOTH, expand=True)
        self.profile_list = ttk.Treeview(
            table,
            columns=("user_id", "full_name", "email"),
            show="headings",
            selectmode="browse",
        )
        for key, title, width in (
            ("user_id", "ID", 86),
            ("full_name", "Name", 150),
            ("email", "Email", 180),
        ):
            self.profile_list.heading(key, text=title)
            self.profile_list.column(key, width=width, minwidth=70, anchor=tk.W)
        scrollbar = ttk.Scrollbar(table, orient=tk.VERTICAL,
                                  command=self.profile_list.yview)
        self.profile_list.configure(yscrollcommand=scrollbar.set)
        self.profile_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.profile_list.bind("<<TreeviewSelect>>", self._select_profile)
        ttk.Button(
            list_card,
            text="＋ New profile",
            style="Primary.TButton",
            command=self._new_profile,
        ).pack(fill=tk.X, pady=(12, 0))

        form = ttk.Frame(content, style="Card.TFrame", padding=18)
        form.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        form.columnconfigure(0, weight=1)
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Profile details", font=("Segoe UI", 13, "bold"),
                  background="#ffffff").grid(
                      row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 12))
        self.user_id_entry = self._add_field(form, "User ID", self.user_id, 1, 0)
        for index, (label, key) in enumerate(PROFILE_FIELDS):
            row = 2 + index // 2
            column = index % 2
            self._add_field(form, label, self.values[key], row, column)
        address_row = 2 + (len(PROFILE_FIELDS) + 1) // 2
        ttk.Label(form, text="Address", background="#ffffff").grid(
            row=address_row, column=0, columnspan=2, sticky=tk.W, pady=(10, 4))
        self.address_text = tk.Text(
            form,
            height=4,
            wrap=tk.WORD,
            relief=tk.SOLID,
            borderwidth=1,
            font=("Segoe UI", 10),
        )
        self.address_text.grid(
            row=address_row + 1, column=0, columnspan=2, sticky="ew")

        photo_row = address_row + 2
        photo_area = ttk.Frame(form, style="Card.TFrame")
        photo_area.grid(row=photo_row, column=0, columnspan=2, sticky="ew", pady=12)
        photo_area.columnconfigure(1, weight=1)
        self.photo_label = ttk.Label(
            photo_area,
            text="No photo selected",
            anchor=tk.CENTER,
            width=22,
            background="#e9eff6",
        )
        self.photo_label.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 12))
        photo_buttons = ttk.Frame(photo_area, style="Card.TFrame")
        photo_buttons.grid(row=0, column=1, sticky="w")
        ttk.Button(photo_buttons, text="Upload photo", command=self._upload_photo).pack(
            side=tk.LEFT, padx=(0, 8))
        ttk.Button(photo_buttons, text="Take photo", command=self._capture_photo).pack(
            side=tk.LEFT)
        ttk.Label(
            photo_area,
            text="Photos remain in local profile storage.",
            style="Subtitle.TLabel",
            background="#ffffff",
        ).grid(row=1, column=1, sticky="w", pady=(8, 0))
        self.photo_status = tk.StringVar(value="Optional")
        ttk.Label(form, textvariable=self.photo_status, background="#ffffff").grid(
            row=photo_row + 1, column=0, columnspan=2, sticky=tk.W)

        action_row = ttk.Frame(form, style="Card.TFrame")
        action_row.grid(row=photo_row + 2, column=0, columnspan=2,
                        sticky="ew", pady=(12, 0))
        ttk.Button(action_row, text="Save profile", style="Primary.TButton",
                   command=self._save_profile).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(action_row, text="Export", command=self._export_profile).pack(
            side=tk.LEFT, padx=(0, 8))
        ttk.Button(action_row, text="Delete", command=self._delete_profile).pack(
            side=tk.LEFT)

    @staticmethod
    def _add_field(
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        row: int,
        column: int,
    ) -> ttk.Entry:
        cell = ttk.Frame(parent, style="Card.TFrame")
        cell.grid(row=row, column=column, sticky="ew", padx=(0, 12), pady=5)
        cell.columnconfigure(0, weight=1)
        ttk.Label(cell, text=label, background="#ffffff").grid(
            row=0, column=0, sticky=tk.W, pady=(0, 4))
        entry = ttk.Entry(cell, textvariable=variable)
        entry.grid(row=1, column=0, sticky="ew")
        return entry

    def _refresh_profiles(self) -> None:
        for item in self.profile_list.get_children():
            self.profile_list.delete(item)
        for profile in self.controller.search(self.search_text.get()):
            self.profile_list.insert(
                "",
                tk.END,
                iid=profile.user_id,
                values=(profile.user_id, profile.full_name, profile.email),
            )

    def _select_profile(self, _event: object) -> None:
        selection = self.profile_list.selection()
        if not selection:
            return
        profile = self.repository.get(selection[0])
        if profile is None:
            return
        self.current_user_id = profile.user_id
        self.user_id.set(profile.user_id)
        self.user_id_entry.configure(state="readonly")
        for _, key in PROFILE_FIELDS:
            self.values[key].set(getattr(profile, key))
        self.address_text.delete("1.0", tk.END)
        self.address_text.insert("1.0", profile.address)
        self.photo_source = Path(profile.photo_reference) if profile.photo_reference else None
        if self.photo_source is not None and not self.photo_source.is_absolute():
            self.photo_source = (
                self.repository.database_path.parent / self.photo_source
            ).resolve()
        self._show_photo(self.photo_source)

    def _new_profile(self) -> None:
        self.current_user_id = None
        self.photo_source = None
        self.user_id_entry.configure(state="normal")
        self.user_id.set("")
        for variable in self.values.values():
            variable.set("")
        self.address_text.delete("1.0", tk.END)
        self.photo_status.set("Optional")
        self._show_photo(None)
        self.profile_list.selection_remove(self.profile_list.selection())

    def _build_profile(self) -> DriverProfile:
        values = {key: variable.get().strip() for key, variable in self.values.items()}
        return DriverProfile(
            user_id=self.user_id.get().strip(),
            **values,
            address=self.address_text.get("1.0", tk.END).strip(),
            photo_reference=str(self.photo_source) if self.photo_source else "",
        )

    def _save_profile(self) -> None:
        profile = self._build_profile()
        errors = profile.validate()
        if errors:
            messagebox.showerror("Check profile details", "\n".join(errors.values()))
            return
        try:
            self.controller.save(profile, is_new=self.current_user_id is None)
        except (ValueError, KeyError, OSError) as error:
            messagebox.showerror("Profile not saved", str(error))
            return
        self.current_user_id = profile.user_id
        self.user_id_entry.configure(state="readonly")
        self._refresh_profiles()
        self.profile_list.selection_set(profile.user_id)
        messagebox.showinfo("Profile saved", "The driver profile was saved locally.")

    def _delete_profile(self) -> None:
        if self.current_user_id is None:
            messagebox.showwarning("No profile selected", "Select a saved profile first.")
            return
        if not messagebox.askyesno(
            "Delete profile",
            f"Delete profile {self.current_user_id}? A managed photo will also be removed.",
        ):
            return
        try:
            self.controller.delete(self.current_user_id, confirmed=True)
        except (KeyError, OSError, PermissionError) as error:
            messagebox.showerror("Profile not deleted", str(error))
            return
        self._new_profile()
        self._refresh_profiles()

    def _upload_photo(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose a driver photo",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"),),
        )
        if selected:
            source = Path(selected).resolve()
            photo_dir = self.repository.managed_photo_dir or (
                self.repository.database_path.parent / "driver_photos"
            )
            photo_dir.mkdir(parents=True, exist_ok=True)
            safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", self.user_id.get().strip())
            suffix = source.suffix.lower() or ".jpg"
            destination = photo_dir / (
                f"{safe_id or 'driver'}-{uuid.uuid4().hex[:8]}{suffix}"
            )
            shutil.copy2(source, destination)
            self.photo_source = destination.resolve()
            self.photo_status.set(f"Selected: {self.photo_source.name}")
            self._show_photo(self.photo_source)

    def _capture_photo(self) -> None:
        try:
            import cv2
        except ImportError as error:
            messagebox.showerror("Camera unavailable", str(error))
            return
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            capture.release()
            messagebox.showerror("Camera unavailable", "Could not open camera 0.")
            return
        captured_frame = None
        window_created = False
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera failed to return a frame.")
                cv2.imshow("Capture driver photo - SPACE to save, ESC to cancel", frame)
                window_created = True
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    return
                if key == 32:
                    captured_frame = frame.copy()
                    break
        except RuntimeError as error:
            messagebox.showerror("Photo capture failed", str(error))
            return
        finally:
            capture.release()
            if window_created:
                cv2.destroyWindow("Capture driver photo - SPACE to save, ESC to cancel")
        if captured_frame is None:
            return
        user_id = self.user_id.get().strip()
        safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", user_id) or "driver"
        photo_dir = self.repository.managed_photo_dir or (
            self.repository.database_path.parent / "driver_photos"
        )
        photo_dir.mkdir(parents=True, exist_ok=True)
        photo_path = photo_dir / f"{safe_id}-{uuid.uuid4().hex[:8]}.jpg"
        if not cv2.imwrite(str(photo_path), captured_frame):
            raise OSError(f"Could not write captured photo to {photo_path}")
        self.photo_source = photo_path.resolve()
        self.photo_status.set(f"Captured: {photo_path.name}")
        self._show_photo(self.photo_source)

    def _show_photo(self, path: Path | None) -> None:
        self.photo_preview = None
        if path is None or not path.is_file():
            self.photo_label.configure(image="", text="No photo selected")
            if path is not None:
                self.photo_status.set("The saved photo file is missing.")
            return
        try:
            from PIL import Image, ImageTk

            image = Image.open(path)
            image.thumbnail((150, 150))
            self.photo_preview = ImageTk.PhotoImage(image)
        except ImportError:
            self.photo_label.configure(image="", text=path.name)
            self.photo_status.set("Photo selected; install Pillow to preview it.")
            return
        except OSError as error:
            self.photo_label.configure(image="", text="Preview unavailable")
            self.photo_status.set(f"Could not preview photo: {error}")
            return
        self.photo_label.configure(image=self.photo_preview, text="")

    def _export_profile(self) -> None:
        if self.current_user_id is None:
            messagebox.showwarning("No profile selected", "Select a saved profile first.")
            return
        profile = self.repository.get(self.current_user_id)
        if profile is None:
            messagebox.showerror("Profile not found", "The selected profile no longer exists.")
            return
        destination = filedialog.asksaveasfilename(
            title="Export profile",
            defaultextension=".json",
            filetypes=(("JSON file", "*.json"),),
            initialfile=f"driver_profile_{profile.user_id}.json",
        )
        if not destination:
            return
        try:
            Path(destination).write_text(
                json.dumps(asdict(profile), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            messagebox.showerror("Export failed", str(error))
            return
        messagebox.showinfo("Profile exported", "The profile was exported to the selected file.")


def run_profile_manager() -> None:
    paths = ProjectPaths.discover()
    root = tk.Tk()
    repository = DriverProfileRepository(
        paths.driver_database,
        managed_photo_dir=paths.driver_photos,
    )
    DriverProfileApp(root, repository)
    root.mainloop()
