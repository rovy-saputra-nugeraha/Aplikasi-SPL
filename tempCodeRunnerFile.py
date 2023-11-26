     logo_path = "logo.jpg"
        try:
            logo_image = PhotoImage(file=logo_path)
            logo_label = tk.Label(main_frame, image=logo_image)
            logo_label.grid(row=0, column=2, rowspan=2, padx=10, pady=5, sticky=tk.E)
        except Exception as e:
            print(f"Error loading logo: {e}")
