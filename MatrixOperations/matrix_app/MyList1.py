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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    matrix1_str = request.form.get('matrix1')
    matrix2_str = request.form.get('matrix2')
    scalar_str = request.form.get('scalar')
    operation = request.form.get('operation')

    errors = []
    
    # Matrix Parsing
    try:
        matrix1 = parse_matrix(matrix1_str)
    except ValueError as e:
        errors.append(f"Matrix 1 error: {str(e)}")
        matrix1 = None

    try:
        matrix2 = parse_matrix(matrix2_str) if matrix2_str else None
    except ValueError as e:
        errors.append(f"Matrix 2 error: {str(e)}")
        matrix2 = None

    # Parse scalar safely
    try:
        scalar = float(scalar_str) if scalar_str else None
    except ValueError:
        scalar = None
        errors.append("Invalid scalar value. Please provide a valid number.")

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
                result = np.transpose(matrix1)
            elif operation == "scalar_multiply":
                if scalar is not None:
                    result = matrix1 * scalar
                else:
                    errors.append("Scalar value is required for scalar multiplication.")
            elif operation == "determinant":
                result = np.linalg.det(matrix1)
            elif operation == "inverse":
                result = np.linalg.inv(matrix1)
            elif operation == "eigenvalues":
                result = np.linalg.eigvals(matrix1)
            elif operation == "eigenvectors":
                result = np.linalg.eig(matrix1)
        except Exception as e:
            errors.append(str(e))

        if isinstance(result, np.ndarray):
            result_str = np.array2string(result, precision=2, separator=', ')
        else:
            result_str = str(result)
    else:
        result_str = None

    return render_template('result.html', result=result_str, errors=errors)


def parse_matrix(matrix_str):
    try:
        return np.array(ast.literal_eval(matrix_str))
    except (ValueError, SyntaxError):
        raise ValueError("Invalid matrix format. Ensure you use a 2D list, e.g., [[1, 2], [3, 4]].")


if __name__ == '__main__':
    app.run(debug=True)
