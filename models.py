from datetime import datetime
from app import db

class MatrixCalculation(db.Model):
    __tablename__ = 'matrix_calculation'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(36), nullable=True)  # ADD THIS
    matrix1 = db.Column(db.JSON, nullable=False)
    matrix2 = db.Column(db.JSON, nullable=True)
    scalar = db.Column(db.Float, nullable=True)
    operation = db.Column(db.String(50), nullable=False)
    result = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<MatrixCalculation {self.operation} at {self.created_at}>'
