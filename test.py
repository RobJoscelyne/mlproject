# Test Script
from src.utils import load_object

preprocessor_path = r'C:\Users\robjo\mlproject\artifacts\proprocessor.pkl'
preprocessor = load_object(preprocessor_path)
print(preprocessor)


