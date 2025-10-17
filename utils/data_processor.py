import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple
import re

class DataProcessor:
    """Handle data processing and validation for fleet analysis"""
    
    def __init__(self):
        self.required_columns = ['make', 'model', 'year', 'daily_mileage']
        self.optional_columns = ['vehicle_type', 'current_mpg', 'fuel_type']
        self.vehicle_types = [
            'Passenger Car', 'SUV', 'Pickup Truck', 'Van/Minivan', 
            'Sedan', 'Hatchback', 'Coupe', 'Convertible', 'Wagon'
        ]
    
    def validate_upload_file(self, uploaded_file) -> Tuple[bool, str]:
        """Validate uploaded file format and structure"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                return False, "Unsupported file format. Please upload CSV or Excel files."
            
            # Check if file is empty
            if df.empty:
                return False, "The uploaded file is empty."
            
            # Check required columns
            missing_columns = []
            for col in self.required_columns:
                if col not in df.columns:
                    # Try to find similar column names
                    similar_cols = self._find_similar_columns(col, df.columns)
                    if not similar_cols:
                        missing_columns.append(col)
            
            if missing_columns:
                return False, f"Missing required columns: {', '.join(missing_columns)}"
            
            # Basic data validation
            validation_errors = self._validate_data_content(df)
            if validation_errors:
                return False, f"Data validation errors: {'; '.join(validation_errors)}"
            
            return True, "File validation successful"
        
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
    
    def process_upload_file(self, uploaded_file) -> pd.DataFrame:
        """Process uploaded file and return standardized DataFrame"""
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Clean and validate data
            df = self._clean_data(df)
            
            # Add derived columns
            df = self._add_derived_columns(df)
            
            return df
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return pd.DataFrame()
    
    def _find_similar_columns(self, target_col: str, available_cols: List[str]) -> List[str]:
        """Find columns with similar names"""
        similar = []
        target_lower = target_col.lower()
        
        for col in available_cols:
            col_lower = col.lower()
            
            # Exact match
            if target_lower == col_lower:
                return [col]
            
            # Partial matches
            if target_lower in col_lower or col_lower in target_lower:
                similar.append(col)
            
            # Common variations
            variations = {
                'daily_mileage': ['mileage', 'miles', 'daily_miles', 'miles_per_day'],
                'vehicle_type': ['type', 'category', 'class', 'vehicle_class'],
                'current_mpg': ['mpg', 'fuel_economy', 'efficiency'],
                'fuel_type': ['fuel']
            }
            
            if target_col in variations:
                for variation in variations[target_col]:
                    if variation in col_lower:
                        similar.append(col)
        
        return similar
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match expected format"""
        column_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower().strip()
            
            # Map common variations to standard names
            if col_lower in ['make', 'manufacturer', 'brand']:
                column_mapping[col] = 'make'
            elif col_lower in ['model', 'vehicle_model']:
                column_mapping[col] = 'model'
            elif col_lower in ['year', 'model_year', 'yr']:
                column_mapping[col] = 'year'
            elif col_lower in ['daily_mileage', 'mileage', 'miles', 'daily_miles', 'miles_per_day']:
                column_mapping[col] = 'daily_mileage'
            elif col_lower in ['vehicle_type', 'type', 'category', 'class', 'vehicle_class']:
                column_mapping[col] = 'vehicle_type'
            elif col_lower in ['current_mpg', 'mpg', 'fuel_economy', 'efficiency']:
                column_mapping[col] = 'current_mpg'
            elif col_lower in ['fuel_type', 'fuel']:
                column_mapping[col] = 'fuel_type'
        
        return df.rename(columns=column_mapping)
    
    def _validate_data_content(self, df: pd.DataFrame) -> List[str]:
        """Validate data content"""
        errors = []
        
        # Check year range
        if 'year' in df.columns:
            invalid_years = df[
                (df['year'] < 1990) | (df['year'] > 2025) | pd.isna(df['year'])
            ]
            if not invalid_years.empty:
                errors.append(f"{len(invalid_years)} rows have invalid years")
        
        # Check daily mileage
        if 'daily_mileage' in df.columns:
            invalid_mileage = df[
                (df['daily_mileage'] < 0) | (df['daily_mileage'] > 1000) | pd.isna(df['daily_mileage'])
            ]
            if not invalid_mileage.empty:
                errors.append(f"{len(invalid_mileage)} rows have invalid daily mileage")
        
        # Check for empty required fields
        for col in self.required_columns:
            if col in df.columns:
                empty_values = df[df[col].isna() | (df[col] == '')]
                if not empty_values.empty:
                    errors.append(f"{len(empty_values)} rows have empty {col}")
        
        return errors
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize data"""
        # Clean text fields
        text_columns = ['make', 'model', 'vehicle_type', 'fuel_type']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
        
        # Convert numeric fields
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        
        if 'daily_mileage' in df.columns:
            df['daily_mileage'] = pd.to_numeric(df['daily_mileage'], errors='coerce')
        
        if 'current_mpg' in df.columns:
            df['current_mpg'] = pd.to_numeric(df['current_mpg'], errors='coerce')
        
        # Remove rows with invalid required data
        initial_rows = len(df)
        df = df.dropna(subset=[col for col in self.required_columns if col in df.columns])
        
        if len(df) < initial_rows:
            st.warning(f"Removed {initial_rows - len(df)} rows due to missing required data")
        
        return df
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns for analysis"""
        # Annual mileage
        if 'daily_mileage' in df.columns:
            df['annual_mileage'] = df['daily_mileage'] * 365
        
        # Vehicle age
        if 'year' in df.columns:
            current_year = 2025
            df['vehicle_age'] = current_year - df['year']
        
        # Standardize vehicle types
        if 'vehicle_type' in df.columns:
            df['vehicle_type_standardized'] = df['vehicle_type'].apply(self._standardize_vehicle_type)
        
        # Add unique identifier
        df['vehicle_id'] = range(1, len(df) + 1)
        
        return df
    
    def _standardize_vehicle_type(self, vehicle_type: str) -> str:
        """Standardize vehicle type categories"""
        if pd.isna(vehicle_type):
            return 'Unknown'
        
        vehicle_type = str(vehicle_type).lower().strip()
        
        # Mapping rules
        if any(word in vehicle_type for word in ['suv', 'sport utility', 'crossover']):
            return 'SUV'
        elif any(word in vehicle_type for word in ['truck', 'pickup']):
            return 'Pickup Truck'
        elif any(word in vehicle_type for word in ['van', 'minivan']):
            return 'Van/Minivan'
        elif any(word in vehicle_type for word in ['sedan', 'saloon']):
            return 'Sedan'
        elif any(word in vehicle_type for word in ['hatchback', 'hatch']):
            return 'Hatchback'
        elif any(word in vehicle_type for word in ['coupe', 'coupé']):
            return 'Coupe'
        elif any(word in vehicle_type for word in ['convertible', 'cabrio']):
            return 'Convertible'
        elif any(word in vehicle_type for word in ['wagon', 'estate']):
            return 'Wagon'
        else:
            return 'Passenger Car'
    
    def create_sample_template(self) -> pd.DataFrame:
        """Create sample template for CSV upload"""
        sample_data = {
            'make': ['Toyota', 'Ford', 'Chevrolet', 'Honda', 'BMW'],
            'model': ['Camry', 'F-150', 'Malibu', 'Accord', 'X3'],
            'year': [2020, 2019, 2021, 2018, 2022],
            'daily_mileage': [45, 120, 30, 60, 25],
            'vehicle_type': ['Sedan', 'Pickup Truck', 'Sedan', 'Sedan', 'SUV'],
            'current_mpg': [32, 20, 29, 30, 26],
            'fuel_type': ['Gasoline', 'Gasoline', 'Gasoline', 'Gasoline', 'Gasoline']
        }
        
        return pd.DataFrame(sample_data)
    
    def export_recommendations(self, recommendations_df: pd.DataFrame, format_type: str = 'csv') -> bytes:
        """Export recommendations to specified format"""
        try:
            if format_type.lower() == 'csv':
                return recommendations_df.to_csv(index=False).encode('utf-8')
            elif format_type.lower() in ['xlsx', 'excel']:
                import io
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    recommendations_df.to_excel(writer, sheet_name='EV_Recommendations', index=False)
                return output.getvalue()
            else:
                raise ValueError("Unsupported export format")
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")
            return b""
