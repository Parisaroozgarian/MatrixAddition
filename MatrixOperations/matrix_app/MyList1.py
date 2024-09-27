from flask import Flask, render_template, request
import numpy as np
import ast

app = Flask(__name__)

# Matrix Class Definition
class MyList:
    def __init__(self, n, m, data=None):
        if data:
            self.__list = [row[:] for row in data]
            self.sizerow = len(self.__list)
            self.sizecol = len(self.__list[0]) if self.sizerow > 0 else 0
        else:
            self.__list = [[0 for _ in range(m)] for _ in range(n)]
            self.sizerow = n
            self.sizecol = m

    def set(self, i, j, x):
        self.__list[i][j] = x

    def __str__(self):
        return '\n'.join(['\t'.join(map(str, row)) for row in self.__list])

    def __add__(self, m2):
        if self.sizerow != m2.sizerow or self.sizecol != m2.sizecol:
            raise ValueError("Matrices must have the same dimensions to be added.")
        l = MyList(self.sizerow, self.sizecol)
        for i in range(self.sizerow):
            for j in range(self.sizecol):
                l.set(i, j, self.__list[i][j] + m2.__list[i][j])
        return l

    def __sub__(self, m2):
        if self.sizerow != m2.sizerow or self.sizecol != m2.sizecol:
            raise ValueError("Matrices must have the same dimensions to be subtracted.")
        l = MyList(self.sizerow, self.sizecol)
        for i in range(self.sizerow):
            for j in range(self.sizecol):
                l.set(i, j, self.__list[i][j] - m2.__list[i][j])
        return l

    def __mul__(self, m2):
        if self.sizerow != m2.sizerow or self.sizecol != m2.sizecol:
            raise ValueError("Matrices must have the same dimensions for element-wise multiplication.")
        l = MyList(self.sizerow, self.sizecol)
        for i in range(self.sizerow):
            for j in range(self.sizecol):
                l.set(i, j, self.__list[i][j] * m2.__list[i][j])
        return l

    def __matmul__(self, m2):
        if self.sizecol != m2.sizerow:
            raise ValueError("Matrices are not aligned for matrix multiplication.")
        l = MyList(self.sizerow, m2.sizecol)
        for i in range(self.sizerow):
            for j in range(m2.sizecol):
                sum = 0
                for k in range(self.sizecol):
                    sum += self.__list[i][k] * m2.__list[k][j]
                l.set(i, j, sum)
        return l

    def transpose(self):
        transposed_list = MyList(self.sizecol, self.sizerow)
        for i in range(self.sizerow):
            for j in range(self.sizecol):
                transposed_list.set(j, i, self.__list[i][j])
        return transposed_list

    def scalar_multiply(self, scalar):
        result = MyList(self.sizerow, self.sizecol)
        for i in range(self.sizerow):
            for j in range(self.sizecol):
                result.set(i, j, self.__list[i][j] * scalar)
        return result

    def to_numpy(self):
        return np.array(self.__list)

    def is_square(self):
        return self.sizerow == self.sizecol  # New method to check if the matrix is square

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    errors = []
    
    matrix1_str = request.form.get('matrix1')
    matrix2_str = request.form.get('matrix2')
    scalar_str = request.form.get('scalar')
    operation = request.form.get('operation')

    print(f"Received Matrix 1: {matrix1_str}")
    print(f"Received Matrix 2: {matrix2_str}")
    print(f"Received Scalar: {scalar_str}")
    print(f"Selected Operation: {operation}")

    # Initialize matrices
    matrix1 = None
    matrix2 = None

    # Matrix Parsing
    try:
        matrix1 = parse_matrix(matrix1_str)
    except ValueError as e:
        errors.append(f"Matrix 1 error: {str(e)}")

    if matrix2_str:
        try:
            matrix2 = parse_matrix(matrix2_str)
        except ValueError as e:
            errors.append(f"Matrix 2 error: {str(e)}")

    # Parse scalar safely
    scalar = None
    if scalar_str:
        try:
            scalar = float(scalar_str)
        except ValueError:
            errors.append("Invalid scalar value. Please provide a valid number.")

    # Prepare result variable
    result = None

    if operation in ["add", "subtract", "multiply", "matmul"] and matrix2 is None:
        errors.append("Matrix 2 is required for the selected operation.")

    if not errors:
        try:
            if operation == "add":
                result = matrix1 + matrix2
            elif operation == "subtract":
                result = matrix1 - matrix2
            elif operation == "multiply":
                result = matrix1 * matrix2
            elif operation == "matmul":
                result = matrix1 @ matrix2
            elif operation == "transpose":
                result = matrix1.transpose()
            elif operation == "scalar_multiply":
                if scalar is not None:
                    result = matrix1.scalar_multiply(scalar)
                else:
                    errors.append("Scalar value is required for scalar multiplication.")
            elif operation == "determinant":
                if matrix1.is_square():
                    result = np.linalg.det(matrix1.to_numpy())
                else:
                    errors.append("Determinant can only be calculated for square matrices.")
            elif operation == "inverse":
                if matrix1.is_square():
                    result = np.linalg.inv(matrix1.to_numpy())
                else:
                    errors.append("Inverse can only be calculated for square matrices.")
            elif operation == "eigenvalues":
                if matrix1.is_square():
                    result = np.linalg.eigvals(matrix1.to_numpy())
                else:
                    errors.append("Eigenvalues can only be calculated for square matrices.")
            elif operation == "eigenvectors":
                if matrix1.is_square():
                    eigenvalues, eigenvectors = np.linalg.eig(matrix1.to_numpy())
                    result = eigenvectors.tolist()  # Store eigenvectors as a list
                else:
                    errors.append("Eigenvectors can only be calculated for square matrices.")

        except Exception as e:
            errors.append("Error in calculation: " + str(e))

    # Prepare the result string
    if isinstance(result, np.ndarray):
        result_str = np.array2string(result, precision=2, separator=', ')
    elif result is not None:
        result_str = str(result)
    else:
        result_str = "No result"

    return render_template('result.html', result=result_str, errors=errors)

def parse_matrix(matrix_str):
    try:
        matrix = ast.literal_eval(matrix_str)
        if not isinstance(matrix, (list, tuple)) or not all(isinstance(row, (list, tuple)) for row in matrix):
            raise ValueError("Matrix must be a list of lists.")
        if len(set(len(row) for row in matrix)) != 1:
            raise ValueError("All rows must have the same number of columns.")
        
        # Calculate number of rows and columns
        n = len(matrix)
        m = len(matrix[0])
        
        return MyList(n, m, matrix)  # Create MyList with rows and columns
    except (ValueError, SyntaxError):
        raise ValueError("Invalid matrix format. Ensure you use a 2D list, e.g., [[1, 2], [3, 4]].")

if __name__ == '__main__':
    app.run(debug=True)
