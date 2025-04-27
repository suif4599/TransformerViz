try:
    from tkinter import Tk, Button, Toplevel, Frame, TclError
    from tkinter.scrolledtext import ScrolledText
    from tkinter import messagebox, font
    import traceback

    def tkinter_show_error(exc_type, exc_value, tb):
        root = Tk()
        root.withdraw()

        available_fonts = set(font.families())
        safe_fonts = [
            'Segoe UI', 
            'Helvetica', 
            'Liberation Sans',
            'Arial',
            'DejaVu Sans'
        ]
        selected_font = next((f for f in safe_fonts if f in available_fonts), 'TkDefaultFont')
        
        top = Toplevel(root)
        top.title(f"{exc_type.__name__}")
        top.geometry("800x600")

        text_font = (selected_font, 24)
        button_font = (selected_font, 18, 'bold')

        main_frame = Frame(top)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        button_frame = Frame(main_frame)
        button_frame.pack(fill='x', pady=(10, 0))
        Button(
            button_frame,
            text="Copy to Clipboard",
            command=lambda: copy_to_clipboard(error_msg, top),
            font=button_font,
            bg='#4a9cff',
            fg='white',
            activebackground='#3b82f6',
            padx=15,
            pady=5
        ).pack(side='left', padx=5)
        Button(
            button_frame,
            text="Close",
            command=top.destroy,
            font=button_font,
            bg='#34d399',
            fg='white',
            activebackground='#10b981',
            padx=15,
            pady=5
        ).pack(side='right', padx=5)
        text_area = ScrolledText(
            main_frame,
            wrap='word',
            font=text_font,
            padx=10,
            pady=10,
            bg='#f8f9fa'
        )
        text_area.pack(fill='both', expand=True)

        error_msg = f"{exc_type.__name__}: {exc_value}\n\n{''.join(traceback.format_exception(exc_type, exc_value, tb))}"
        text_area.insert('end', error_msg)
        text_area.configure(state='disabled')

        top.protocol("WM_DELETE_WINDOW", root.destroy)
        top.grab_set()
        root.wait_window(top)
        try:
            root.destroy()
        except TclError:
            pass

    def copy_to_clipboard(text, window):
        try:
            window.clipboard_clear()
            window.clipboard_append(text)
            messagebox.showinfo("Success", "Traceback copied to clipboard!", parent=window)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy to clipboard: {e}", parent=window)
except ImportError:
    def tkinter_show_error(exc_type, exc_value, tb):
        print(f"Error: {exc_type.__name__}: {exc_value}")
        print("".join(traceback.format_exception(exc_type, exc_value, tb)))
        print("\nTkinter is not available.")