            # Tampilkan hasil
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)

            # Iterasi Gauss-Jordan
            gauss_jordan_solution = gauss_jordan_result[:, -1]
            self.result_text.insert(tk.END, "Iterasi Gauss-Jordan:\n{}\n\n".format(
                np.array2string(gauss_jordan_solution, precision=5, separator=', ')))

            # Iterasi Jacobi
            self.result_text.insert(tk.END, "Iterasi pada Jacobi: Iterasi Ke-{}\n{}\n\n".format(
                jacobi_iterations, np.array2string(jacobi_result, precision=5, separator=', ')))

            # Iterasi Gauss-Seidel
            self.result_text.insert(tk.END, "Iterasi pada Gauss-Seidel: Iterasi Ke-{}\n{}".format(
                gauss_seidel_iterations, np.array2string(gauss_seidel_result, precision=5, separator=', ')))

            self.result_text.config(state=tk.DISABLED)