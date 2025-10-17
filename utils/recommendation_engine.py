import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional
import streamlit as st

class RecommendationEngine:
    """ML-powered EV recommendation engine"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.knn_model = None
        self.ev_database = None
        self.feature_weights = {
            'size_class': 0.3,
            'performance': 0.25,
            'efficiency': 0.2,
            'range': 0.15,
            'year': 0.1
        }
    
    def initialize_ev_database(self, ev_data: List[Dict]) -> bool:
        """Initialize EV database for recommendations"""
        try:
            if not ev_data:
                st.error("No EV data provided for initialization")
                return False
            
            # Convert to DataFrame
            self.ev_database = pd.DataFrame(ev_data)
            
            # Clean and prepare EV data
            self.ev_database = self._prepare_ev_data(self.ev_database)
            
            # Build ML model
            self._build_recommendation_model()
            
            st.success(f"EV database initialized with {len(self.ev_database)} vehicles")
            return True
        
        except Exception as e:
            st.error(f"Error initializing EV database: {str(e)}")
            return False
    
    def get_recommendations(self, vehicle: Dict, top_n: int = 5) -> List[Dict]:
        """Get EV recommendations for a single vehicle"""
        try:
            if self.ev_database is None or self.ev_database.empty:
                return self._get_fallback_recommendations(vehicle, top_n)
            
            # Prepare input vehicle features
            vehicle_features = self._extract_vehicle_features(vehicle)
            
            # Get similarity scores
            similarities = self._calculate_similarities(vehicle_features)
            
            # Rank and filter recommendations
            recommendations = self._rank_recommendations(similarities, vehicle, top_n)
            
            return recommendations
        
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def get_batch_recommendations(self, fleet_data: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        """Generate recommendations for entire fleet"""
        try:
            recommendations = []
            total_vehicles = len(fleet_data)
            
            for idx, row in fleet_data.iterrows():
                if progress_callback:
                    progress_callback(idx, total_vehicles)
                
                # Get recommendations for this vehicle
                vehicle_dict = row.to_dict()
                recs = self.get_recommendations(vehicle_dict, top_n=3)
                
                # Add recommendations to results
                for i, rec in enumerate(recs):
                    rec_row = vehicle_dict.copy()
                    rec_row['recommendation_rank'] = i + 1
                    rec_row.update(rec)
                    recommendations.append(rec_row)
            
            return pd.DataFrame(recommendations)
        
        except Exception as e:
            st.error(f"Error in batch processing: {str(e)}")
            return pd.DataFrame()
    
    def _prepare_ev_data(self, ev_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare EV database"""
        # Remove duplicates
        ev_df = ev_df.drop_duplicates(subset=['make', 'model', 'year'], keep='first')
        
        # Fill missing values
        numeric_columns = ['city08', 'highway08', 'comb08', 'year', 'cylinders', 'displ']
        for col in numeric_columns:
            if col in ev_df.columns:
                ev_df[col] = pd.to_numeric(ev_df[col], errors='coerce')
                ev_df[col] = ev_df[col].fillna(ev_df[col].median())
        
        # Add derived features
        ev_df = self._add_ev_features(ev_df)
        
        return ev_df
    
    def _add_ev_features(self, ev_df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for EV analysis"""
        # Size classification based on vehicle class or model name
        ev_df['size_class'] = ev_df.apply(self._classify_size, axis=1)
        
        # Performance score
        ev_df['performance_score'] = self._calculate_performance_score(ev_df)
        
        # Efficiency score (MPGe)
        ev_df['efficiency_score'] = ev_df['comb08'].fillna(100)  # Default MPGe for EVs
        
        # Range estimate
        if 'ev_range' not in ev_df.columns:
            ev_df['ev_range'] = ev_df['efficiency_score'] * 3.5  # Rough estimation
        
        # Price category (estimated based on make/model)
        ev_df['price_category'] = ev_df.apply(self._estimate_price_category, axis=1)
        
        return ev_df
    
    def _classify_size(self, row: pd.Series) -> int:
        """Classify vehicle size (1=compact, 2=mid-size, 3=full-size, 4=SUV/truck)"""
        model = str(row.get('model', '')).lower()
        vclass = str(row.get('VClass', '')).lower()
        
        # SUV/Truck indicators
        if any(word in model + vclass for word in ['suv', 'truck', 'tahoe', 'suburban', 'escalade', 'x3', 'x5', 'q7']):
            return 4
        
        # Full-size indicators
        if any(word in model + vclass for word in ['s-class', 'a8', '7 series', 'ls', 'continental']):
            return 3
        
        # Compact indicators
        if any(word in model + vclass for word in ['compact', 'mini', 'smart', 'leaf', 'i3']):
            return 1
        
        # Default to mid-size
        return 2
    
    def _calculate_performance_score(self, ev_df: pd.DataFrame) -> pd.Series:
        """Calculate performance score based on available metrics"""
        # Base performance on efficiency and estimated acceleration
        performance = ev_df['comb08'].fillna(100)
        
        # Adjust for luxury/performance brands
        luxury_brands = ['tesla', 'porsche', 'bmw', 'mercedes', 'audi', 'jaguar']
        for brand in luxury_brands:
            mask = ev_df['make'].str.lower().str.contains(brand, na=False)
            performance.loc[mask] += 20
        
        return performance
    
    def _estimate_price_category(self, row: pd.Series) -> int:
        """Estimate price category (1=budget, 2=mid, 3=luxury)"""
        make = str(row.get('make', '')).lower()
        model = str(row.get('model', '')).lower()
        
        luxury_brands = ['tesla', 'porsche', 'bmw', 'mercedes', 'audi', 'jaguar', 'lucid']
        if any(brand in make for brand in luxury_brands):
            return 3
        
        budget_models = ['leaf', 'bolt', 'i3', 'soul']
        if any(model_name in model for model_name in budget_models):
            return 1
        
        return 2
    
    def _extract_vehicle_features(self, vehicle: Dict) -> np.ndarray:
        """Extract features from input vehicle for comparison"""
        features = []
        
        # Size classification (estimated from vehicle type)
        vehicle_type = vehicle.get('vehicle_type_standardized', 'Passenger Car')
        size_mapping = {
            'Passenger Car': 2, 'Sedan': 2, 'Hatchback': 1, 'Coupe': 2,
            'SUV': 4, 'Pickup Truck': 4, 'Van/Minivan': 4, 'Convertible': 2, 'Wagon': 2
        }
        features.append(size_mapping.get(vehicle_type, 2))
        
        # Performance estimate (based on year and type)
        year = vehicle.get('year', 2020)
        base_performance = 80 + (year - 2015) * 2  # Newer = slightly better
        if vehicle_type in ['SUV', 'Pickup Truck']:
            base_performance += 10
        features.append(base_performance)
        
        # Efficiency requirement (based on current MPG if available)
        current_mpg = vehicle.get('current_mpg', 25)
        target_efficiency = current_mpg * 3  # Target 3x efficiency with EV
        features.append(target_efficiency)
        
        # Range requirement (based on daily mileage)
        daily_mileage = vehicle.get('daily_mileage', 40)
        required_range = daily_mileage * 5  # 5 days of driving
        features.append(required_range)
        
        # Year preference
        features.append(year)
        
        return np.array(features).reshape(1, -1)
    
    def _build_recommendation_model(self):
        """Build ML model for similarity matching"""
        try:
            # Extract features from EV database
            features = []
            for _, row in self.ev_database.iterrows():
                feature_vector = [
                    row['size_class'],
                    row['performance_score'],
                    row['efficiency_score'],
                    row['ev_range'],
                    row['year']
                ]
                features.append(feature_vector)
            
            features_array = np.array(features)
            
            # Fit scaler and transform features
            self.scaler.fit(features_array)
            scaled_features = self.scaler.transform(features_array)
            
            # Build KNN model
            self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
            self.knn_model.fit(scaled_features)
            
        except Exception as e:
            st.error(f"Error building recommendation model: {str(e)}")
    
    def _calculate_similarities(self, vehicle_features: np.ndarray) -> List[Tuple[int, float]]:
        """Calculate similarities between input vehicle and EV database"""
        try:
            # Scale input features
            scaled_features = self.scaler.transform(vehicle_features)
            
            # Get nearest neighbors
            distances, indices = self.knn_model.kneighbors(scaled_features)
            
            # Convert distances to similarities (cosine distance -> cosine similarity)
            similarities = [(idx, 1 - dist) for idx, dist in zip(indices[0], distances[0])]
            
            return similarities
        
        except Exception as e:
            st.error(f"Error calculating similarities: {str(e)}")
            return []
    
    def _rank_recommendations(self, similarities: List[Tuple[int, float]], 
                            original_vehicle: Dict, top_n: int) -> List[Dict]:
        """Rank and format recommendations"""
        recommendations = []
        
        for idx, similarity_score in similarities[:top_n]:
            ev_row = self.ev_database.iloc[idx]
            
            recommendation = {
                'ev_make': ev_row['make'],
                'ev_model': ev_row['model'],
                'ev_year': int(ev_row['year']),
                'ev_efficiency': float(ev_row['efficiency_score']),
                'ev_range': float(ev_row['ev_range']),
                'ev_price_category': ev_row['price_category'],
                'similarity_score': float(similarity_score),
                'size_match': self._calculate_size_match(original_vehicle, ev_row),
                'performance_match': self._calculate_performance_match(original_vehicle, ev_row)
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _calculate_size_match(self, original: Dict, ev: pd.Series) -> str:
        """Calculate size match description"""
        orig_type = original.get('vehicle_type_standardized', 'Passenger Car')
        ev_size = ev['size_class']
        
        size_names = {1: 'Compact', 2: 'Mid-size', 3: 'Full-size', 4: 'Large/SUV'}
        
        if orig_type in ['SUV', 'Pickup Truck'] and ev_size == 4:
            return 'Excellent'
        elif orig_type in ['Sedan', 'Passenger Car'] and ev_size == 2:
            return 'Excellent'
        elif orig_type == 'Hatchback' and ev_size == 1:
            return 'Excellent'
        else:
            return 'Good'
    
    def _calculate_performance_match(self, original: Dict, ev: pd.Series) -> str:
        """Calculate performance match description"""
        ev_performance = ev['performance_score']
        
        if ev_performance >= 120:
            return 'High Performance'
        elif ev_performance >= 100:
            return 'Good Performance'
        else:
            return 'Standard Performance'
    
    def _get_fallback_recommendations(self, vehicle: Dict, top_n: int) -> List[Dict]:
        """Provide fallback recommendations when EV database is unavailable"""
        # Basic rule-based recommendations
        vehicle_type = vehicle.get('vehicle_type_standardized', 'Passenger Car')
        
        fallback_evs = {
            'Sedan': [
                {'make': 'Tesla', 'model': 'Model 3', 'year': 2024, 'efficiency': 130, 'range': 310},
                {'make': 'BMW', 'model': 'i4', 'year': 2024, 'efficiency': 116, 'range': 270},
                {'make': 'Polestar', 'model': '2', 'year': 2024, 'efficiency': 107, 'range': 260}
            ],
            'SUV': [
                {'make': 'Tesla', 'model': 'Model Y', 'year': 2024, 'efficiency': 122, 'range': 330},
                {'make': 'Ford', 'model': 'Mustang Mach-E', 'year': 2024, 'efficiency': 101, 'range': 300},
                {'make': 'BMW', 'model': 'iX', 'year': 2024, 'efficiency': 86, 'range': 380}
            ],
            'Pickup Truck': [
                {'make': 'Ford', 'model': 'F-150 Lightning', 'year': 2024, 'efficiency': 76, 'range': 300},
                {'make': 'Rivian', 'model': 'R1T', 'year': 2024, 'efficiency': 70, 'range': 350},
                {'make': 'Chevrolet', 'model': 'Silverado EV', 'year': 2024, 'efficiency': 85, 'range': 400}
            ]
        }
        
        # Get recommendations for vehicle type
        type_recs = fallback_evs.get(vehicle_type, fallback_evs['Sedan'])
        
        recommendations = []
        for i, ev in enumerate(type_recs[:top_n]):
            rec = {
                'ev_make': ev['make'],
                'ev_model': ev['model'],
                'ev_year': ev['year'],
                'ev_efficiency': ev['efficiency'],
                'ev_range': ev['range'],
                'ev_price_category': 2,  # Mid-range
                'similarity_score': 0.8 - (i * 0.1),  # Decreasing similarity
                'size_match': 'Good',
                'performance_match': 'Good Performance'
            }
            recommendations.append(rec)
        
        return recommendations
