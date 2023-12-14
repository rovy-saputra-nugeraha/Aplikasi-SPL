# Mengimpor pustaka yang diperlukan
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Fungsi untuk eliminasi Gauss-Jordan
def gauss_jordan_elimination(matrix):
    # Implementasi algoritma eliminasi Gauss-Jordan
    baris, kolom = len(matrix), len(matrix[0])

    # Loop melalui setiap baris
    for i in range(baris):
        pivot = matrix[i][i]

        # Skala baris saat ini untuk membuat pivot menjadi 1
        for j in range(kolom):
            matrix[i][j] /= pivot

        # Hapus baris lain
        for k in range(baris):
            if k != i:
                faktor = matrix[k][i]
                for j in range(kolom):
                    matrix[k][j] -= faktor * matrix[i][j]

    return matrix

# Fungsi untuk metode iterasi Jacobi
def jacobi_iteration(A, b, x0, max_iterations=50, tolerance=1e-3):
    # Implementasi metode iterasi Jacobi untuk menyelesaikan sistem persamaan linear
    n = len(A)
    x = np.copy(x0)

    # Iterasi untuk jumlah maksimum iterasi
    for iterasi in range(max_iterations):
        x_lama = np.copy(x)
        # Perbarui setiap komponen vektor solusi
        for i in range(n):
            sigma = np.dot(A[i, :n], x_lama) - A[i, i] * x_lama[i]
            x[i] = (b[i] - sigma) / A[i, i]

        # Periksa konvergensi
        if np.linalg.norm(x - x_lama, ord=np.inf) < tolerance:
            return x, iterasi + 1  # Mengembalikan solusi dan jumlah iterasi

    return x, max_iterations  # Mengembalikan solusi dan jumlah iterasi maksimum

# Fungsi untuk metode iterasi Gauss-Seidel
def gauss_seidel_iteration(A, b, x0, max_iterations=50, tolerance=1e-3):
    # Implementasi metode iterasi Gauss-Seidel untuk menyelesaikan sistem persamaan linear
    n = len(A)
    x = np.copy(x0)

    # Iterasi untuk jumlah maksimum iterasi
    for iterasi in range(1, max_iterations + 1):
        x_lama = np.copy(x)
        # Perbarui setiap komponen vektor solusi menggunakan komponen yang sudah diperbarui
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_lama[i+1:])
            x[i] = (b[i] - sigma) / A[i, i]

        # Periksa konvergensi
        if np.linalg.norm(x - x_lama, ord=np.inf) < tolerance:
            return x, iterasi  # Mengembalikan solusi dan jumlah iterasi

    return None, max_iterations  # Mengembalikan None jika tidak konvergen, dan jumlah iterasi maksimum

# Kelas untuk aplikasi GUI
class LinearEquationSolverGUI:
    def __init__(self, root):
        # Inisialisasi GUI dengan jendela root yang diberikan
        self.root = root
        self.root.title("Aplikasi Persamaan Linear")  # Tentukan judul jendela

        # Muat dan ubah ukuran gambar logo
        try:
            logo_image = Image.open("logo.png")  # Ganti dengan path logo 
            logo_image = logo_image.resize((100, 100))  # Sesuaikan ukuran jika diperlukan
            self.logo = ImageTk.PhotoImage(logo_image)
        except Exception as e:
            print("Error loading or resizing the logo:", e)
            self.logo = None

        # Buat widget untuk GUI
        self.create_widgets()

    def create_widgets(self):
        # Buat frame utama
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0)

        # Entri matriks A
        ttk.Label(main_frame, text="Matriks A:", style="Bold.TLabel").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        self.matrix_a_entry = tk.Text(main_frame, width=30, height=5)
        self.matrix_a_entry.grid(row=0, column=1, padx=10, pady=5)

        # Entri vektor b
        ttk.Label(main_frame, text="Vektor b:", style="Bold.TLabel").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        self.vector_b_entry = tk.Text(main_frame, width=30, height=2)
        self.vector_b_entry.grid(row=1, column=1, padx=10, pady=5)

        # Tombol "Cari"
        solve_button = ttk.Button(main_frame, text="Cari", command=self.solve, style="Accent.TButton")
        solve_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Tampilkan logo
        if self.logo:
            logo_label = tk.Label(main_frame, image=self.logo)
            logo_label.grid(row=0, column=2, rowspan=2, padx=10, pady=5, sticky=tk.E)

        # Widget teks hasil
        self.result_text = tk.Text(main_frame, height=10, width=40, state=tk.DISABLED, wrap=tk.WORD, font=('Helvetica', 10))
        self.result_text.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

        # Konfigurasi gaya
        self.style = ttk.Style()
        self.style.configure("Accent.TButton",
                            background="#0000FF",
                            foreground="black",
                            padding=(10, 5),
                            font=('Helvetica', 12, 'bold'))

        # Gaya tambahan untuk label dan teks hasil
        self.style.configure("Bold.TLabel", font=('Helvetica', 12, 'bold'))
        self.style.configure("Result.TText", font=('Helvetica', 10))

        # Gaya hover untuk tombol "Cari"
        self.style.map("Accent.TButton",
                    background=[("active", "green")],
                    foreground=[("active", "green")])

    def solve(self):
        # Dapatkan nilai input
        matrix_a_str = self.matrix_a_entry.get("1.0", tk.END)
        vector_b_str = self.vector_b_entry.get("1.0", tk.END)

        try:
            # Parse input ke array numpy
            matrix_a = np.array(eval(matrix_a_str), dtype=float)
            vector_b = np.array(eval(vector_b_str), dtype=float)

            # Solusi awal
            x0 = np.zeros_like(vector_b, dtype=float)

            # Gauss-Jordan
            augmented_matrix = np.column_stack((matrix_a, vector_b))
            gauss_jordan_result = gauss_jordan_elimination(augmented_matrix)

            # Jacobi
            jacobi_result, jacobi_iterations = jacobi_iteration(matrix_a, vector_b, x0)

            # Gauss-Seidel
            gauss_seidel_result, gauss_seidel_iterations = gauss_seidel_iteration(matrix_a, vector_b, x0)
            
            # Tampilkan hasil
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)

            # Iterasi Gauss-Jordan
            gauss_jordan_solution = gauss_jordan_result[:, -1]
            gauss_jordan_format = "[{:0.3f}, {:0.3f}, {:0.3f}]"

            self.result_text.insert(tk.END, "Iterasi Gauss-Jordan:\n" + gauss_jordan_format.format(*gauss_jordan_solution) + "\n\n")

            # Iterasi Jacobi
            jacobi_format = "Iterasi pada Jacobi: Iterasi Ke-{}\n[{:0.3f}, {:0.3f}, {:0.3f}]\n\n"
            self.result_text.insert(tk.END, jacobi_format.format(jacobi_iterations, *jacobi_result))

            # Iterasi Gauss-Seidel
            gauss_seidel_format = "Iterasi pada Gauss-Seidel: Iterasi Ke-{}\n[{:0.3f}, {:0.3f}, {:0.3f}]"
            self.result_text.insert(tk.END, gauss_seidel_format.format(gauss_seidel_iterations, *gauss_seidel_result))

            self.result_text.config(state=tk.DISABLED)

        except Exception as e:
            # Tampilkan error jika input tidak valid
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Error: {}".format(e))
            self.result_text.config(state=tk.DISABLED)

# Program utama
if __name__ == "__main__":
    root = tk.Tk()
    app = LinearEquationSolverGUI(root)
    root.mainloop()