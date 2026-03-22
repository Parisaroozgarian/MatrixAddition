import os
from flask import Flask, render_template, request, session
import uuid
import numpy as np
import ast
import logging
import json
from flask_sqlalchemy import SQLAlchemy


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev")

# Configure SQLAlchemy
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize SQLAlchemy
db = SQLAlchemy()
db.init_app(app)

# Import models after db initialization to avoid circular imports
from models import MatrixCalculation  # noqa: E402

# Create tables
with app.app_context():
    db.create_all()


def matrix_to_json(matrix):
    """Convert numpy matrix to JSON-serializable format."""
    if isinstance(matrix, np.ndarray):
        return matrix.tolist()
    elif isinstance(matrix, (float, np.float64)):
        return float(matrix)
    elif isinstance(matrix, (complex, np.complex128)):
        return str(matrix)
    return matrix


@app.before_request
def set_session_id():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())


@app.template_filter('tojson')
def tojson_filter(obj):
    """Custom tojson filter that handles numpy arrays."""
    return json.dumps(matrix_to_json(obj))


def parse_matrix(matrix_str):
    """Parse a string representation of a matrix into a numpy array."""
    try:
        if not matrix_str or matrix_str.isspace():
            return None

        # Parse string to Python list
        matrix = ast.literal_eval(matrix_str)

        # Validate structure
        if not isinstance(matrix, (list, tuple)) or not all(
                isinstance(row, (list, tuple)) for row in matrix):
            raise ValueError("Matrix must be a list of lists")

        # Convert to numpy array
        return np.array(matrix, dtype=float)

    except (ValueError, SyntaxError) as e:
        raise ValueError(
            f"Invalid matrix format: {str(e)}. Use format like [[1, 2], [3, 4]]")
    except Exception as e:
        raise ValueError(f"Error parsing matrix: {str(e)}")


def validate_matrices(matrix1, matrix2=None, operation=None):
    """Validate matrices based on the operation."""
    if matrix1 is None:
        raise ValueError("First matrix is required")

    if operation in ["add", "subtract", "multiply"]:
        if matrix2 is None:
            raise ValueError("Second matrix is required for this operation")
        if matrix1.shape != matrix2.shape:
            raise ValueError(f"Matrices must have same shape for {operation}")

    elif operation == "matmul":
        if matrix2 is None:
            raise ValueError("Second matrix is required for matrix multiplication")
        if matrix1.shape[1] != matrix2.shape[0]:
            raise ValueError(
                f"Matrix shapes incompatible for multiplication: {matrix1.shape} and {matrix2.shape}")

    elif operation in ["determinant", "inverse", "eigenvalues", "eigenvectors"]:
        if matrix1.shape[0] != matrix1.shape[1]:
            raise ValueError(f"Square matrix required for {operation}")


@app.route('/', methods=['GET', "POST"])
def index():
    result = None
    result_type = None
    errors = []
    recent_calculations = []

    try:
        if request.method == 'POST':
            # Get form inputs
            matrix1_str = request.form.get('matrix1')
            matrix2_str = request.form.get('matrix2')
            scalar_str = request.form.get('scalar')
            operation = request.form.get('operation')

            # Parse first matrix (required for all operations)
            matrix1 = parse_matrix(matrix1_str)
            if matrix1 is None:
                raise ValueError("First matrix is required")

            # Parse second matrix (if provided)
            matrix2 = parse_matrix(matrix2_str) if matrix2_str else None

            # Parse scalar (if provided)
            scalar = float(scalar_str) if scalar_str else None

            # Validate matrices based on operation
            validate_matrices(matrix1, matrix2, operation)

            # Perform calculations
            if operation == "add":
                result = np.add(matrix1, matrix2)
            elif operation == "subtract":
                result = np.subtract(matrix1, matrix2)
            elif operation == "multiply":
                result = np.multiply(matrix1, matrix2)
            elif operation == "matmul":
                result = np.matmul(matrix1, matrix2)
            elif operation == "transpose":
                result = np.transpose(matrix1)
            elif operation == "scalar_multiply":
                if scalar is None:
                    raise ValueError("Scalar value is required for scalar multiplication")
                result = np.multiply(matrix1, scalar)
            elif operation == "determinant":
                result = float(np.linalg.det(matrix1))
            elif operation == "inverse":
                result = np.linalg.inv(matrix1)
            elif operation == "eigenvalues":
                result = np.linalg.eigvals(matrix1)
            elif operation == "eigenvectors":
                eigenvals, eigenvecs = np.linalg.eig(matrix1)
                result = f"Eigenvalues:\n{eigenvals}\n\nEigenvectors:\n{eigenvecs}"

            # Store the type of result
            if operation in ["determinant"]:
                result_type = "scalar"
            elif operation in ["eigenvalues", "eigenvectors"]:
                result_type = "special"
            else:
                result_type = "matrix"

            # Convert result to JSON-serializable format before template rendering
            if result is not None:
                if result_type == "matrix":
                    result = matrix_to_json(result)
                elif result_type == "special":
                    pass
                else:
                    result = float(result)

            logger.debug(f"Operation: {operation}")
            logger.debug(f"Result type: {result_type}")
            logger.debug(f"Result: {result}")

            try:
                # Save calculation to database with user_id
                calc = MatrixCalculation(
                    matrix1=matrix_to_json(matrix1),
                    matrix2=matrix_to_json(matrix2) if matrix2 is not None else None,
                    scalar=scalar,
                    operation=operation,
                    result=matrix_to_json(result),
                    user_id=session['user_id']
                )
                db.session.add(calc)
                db.session.commit()
            except Exception as db_error:
                logger.error(f"Database error: {str(db_error)}")
                pass

    except Exception as e:
        errors.append(str(e))
        logger.error(f"Error in calculation: {str(e)}")

    # Get recent calculations for this user only
    try:
        recent_calculations = MatrixCalculation.query.filter_by(
            user_id=session['user_id']
        ).order_by(MatrixCalculation.created_at.desc()).limit(5).all()
    except Exception as e:
        logger.error(f"Error fetching recent calculations: {str(e)}")
        recent_calculations = []

    return render_template('index.html',
                           result=result,
                           result_type=result_type,
                           errors=errors,
                           recent_calculations=recent_calculations)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)