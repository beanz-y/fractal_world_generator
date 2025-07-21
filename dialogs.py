# beanz-y/fractal_world_generator/fractal_world_generator-28f75751b57dacf83432892d2293f1e3754a3ba6/dialogs.py

import tkinter as tk
from tkinter import ttk, simpledialog
from constants import THEME_NAME_FRAGMENTS
from utils import generate_contextual_name

class PlacemarkDialog(simpledialog.Dialog):
    """A dialog for creating custom placemarks with a name and type."""
    def body(self, master):
        self.title("Place Custom Feature")
        self.parent = self.master
        self.name_var = tk.StringVar()
        
        name_frame = ttk.Frame(master)
        name_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        ttk.Label(name_frame, text="Name:").pack(side=tk.LEFT)
        self.name_entry = ttk.Entry(name_frame, textvariable=self.name_var)
        self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        random_button = ttk.Button(name_frame, text="🎲", width=3, command=self.generate_and_set_name)
        random_button.pack(side=tk.LEFT)
        
        self.result = None
        self.generate_and_set_name()
        return self.name_entry

    def generate_and_set_name(self, event=None):
        theme = self.parent.params['theme'].get()
        name_fragments = THEME_NAME_FRAGMENTS.get(theme, THEME_NAME_FRAGMENTS['High Fantasy'])
        used_names = self.parent.generator.used_names if self.parent.generator else set()
        base_name = generate_contextual_name(name_fragments, used_names)
        self.name_var.set(base_name)

    def apply(self):
        name = self.name_var.get()
        if name:
            if self.parent.generator and name not in self.parent.generator.used_names:
                self.parent.generator.used_names.add(name)
            self.result = {'name': name, 'type': 'placemark'}