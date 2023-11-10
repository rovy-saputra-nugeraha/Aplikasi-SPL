import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def gauss_jordan_elimination(matrix):
    rows, cols = len(matrix), len(matrix[0])

    for i in range(rows):
        pivot = matrix[i][i]
        for j in range(cols):
            matrix[i][j] /= pivot

        for k in range(rows):
            if k != i:
                factor = matrix[k][i]
                for j in range(cols):
                    matrix[k][j] -= factor * matrix[i][j]

    return matrix

def jacobi_iteration(A, b, x0, max_iterations=50, tolerance=1e-6):
    n = len(A)
    x = np.copy(x0)

    for iteration in range(max_iterations):
        x_old = np.copy(x)
        for i in range(n):
            sigma = np.dot(A[i, :n], x_old) - A[i, i] * x_old[i]
            x[i] = (b[i] - sigma) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tolerance:
            break

    return x

def gauss_seidel_iteration(A, b, x0, max_iterations=50, tolerance=1e-6):
    n = len(A)
    x = np.copy(x0)

    for iteration in range(max_iterations):
        x_old = np.copy(x)
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - sigma) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tolerance:
            break

    return x

class LinearEquationSolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Persamaan Linear")

        # Load and resize the logo image
        try:
            logo_image = Image.open("logo.png")  # Replace with the path to your logo image
            logo_image = logo_image.resize((100, 100))  # Adjust the size as needed
            self.logo = ImageTk.PhotoImage(logo_image)
        except Exception as e:
            print("Error loading or resizing the logo:", e)
            self.logo = None

        self.create_widgets()

    def create_widgets(self):
        # Matrix A entry
        ttk.Label(self.root, text="Matrix A:").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        self.matrix_a_entry = tk.Text(self.root, width=30, height=5)
        self.matrix_a_entry.grid(row=0, column=1, padx=10, pady=5)

        # Vector b entry
        ttk.Label(self.root, text="Vector b:").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        self.vector_b_entry = tk.Text(self.root, width=30, height=2)
        self.vector_b_entry.grid(row=1, column=1, padx=10, pady=5)

        # Solve button
        solve_button = ttk.Button(self.root, text="Cari", command=self.solve)
        solve_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Result text widget
        self.result_text = tk.Text(self.root, height=10, width=40, state=tk.DISABLED)
        self.result_text.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

        # Logo display
        if self.logo:
            logo_label = tk.Label(self.root, image=self.logo)
            logo_label.grid(row=0, column=2, rowspan=2, padx=10, pady=5)

        # Style configuration
        self.style = ttk.Style()
        self.style.configure("Accent.TButton", background="#4CAF50", foreground="white", padding=(10, 5), font=('Helvetica', 12, 'bold'))

    def solve(self):
        # Get input values
        matrix_a_str = self.matrix_a_entry.get("1.0", tk.END)
        vector_b_str = self.vector_b_entry.get("1.0", tk.END)

        try:
            # Parse input to numpy arrays
            matrix_a = np.array(eval(matrix_a_str), dtype=float)
            vector_b = np.array(eval(vector_b_str), dtype=float)

            # Initial solution
            x0 = np.zeros_like(vector_b, dtype=float)

            # Gauss-Jordan
            augmented_matrix = np.column_stack((matrix_a, vector_b))
            gauss_jordan_result = gauss_jordan_elimination(augmented_matrix)

            # Jacobi
            jacobi_result = jacobi_iteration(matrix_a, vector_b, x0)

            # Gauss-Seidel
            gauss_seidel_result = gauss_seidel_iteration(matrix_a, vector_b, x0)

            # Display results
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Iterasi Gauss-Jordan:\n{}\n\n".format(gauss_jordan_result[:, -1]))
            self.result_text.insert(tk.END, "Iterasi Jacobi:\n{}\n\n".format(jacobi_result))
            self.result_text.insert(tk.END, "Iterasi Gauss-Seidel:\n{}".format(gauss_seidel_result))
            self.result_text.config(state=tk.DISABLED)

        except Exception as e:
            # Display error if input is invalid
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Error: {}".format(e))
            self.result_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = LinearEquationSolverGUI(root)
    root.mainloop()
