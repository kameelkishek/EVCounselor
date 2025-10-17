import requests
import pandas as pd
import streamlit as st
from typing import List, Dict, Optional
import time
import json

class EPAClient:
    """Client for EPA Fuel Economy API integration"""
    
    def __init__(self):
        self.base_url = "https://www.fueleconomy.gov/ws/rest"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EV-Fleet-Recommendation-Engine/1.0'
        })
    
    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            response = self.session.get(f"{self.base_url}/vehicle/menu/make", timeout=10)
            return response.status_code == 200
        except Exception as e:
            st.error(f"API connection test failed: {str(e)}")
            return False
    
    def get_makes(self) -> List[Dict]:
        """Get all vehicle makes"""
        try:
            response = self.session.get(f"{self.base_url}/vehicle/menu/make")
            response.raise_for_status()
            
            # Parse XML response (EPA API returns XML)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            makes = []
            for make_elem in root.findall('.//value'):
                makes.append({
                    'make': make_elem.text,
                    'text': make_elem.text
                })
            
            return makes
        except Exception as e:
            st.error(f"Error fetching makes: {str(e)}")
            return []
    
    def get_models_for_make(self, make: str) -> List[Dict]:
        """Get models for a specific make"""
        try:
            response = self.session.get(f"{self.base_url}/vehicle/menu/model?make={make}")
            response.raise_for_status()
            
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            models = []
            for model_elem in root.findall('.//value'):
                models.append({
                    'model': model_elem.text,
                    'text': model_elem.text
                })
            
            return models
        except Exception as e:
            st.error(f"Error fetching models for {make}: {str(e)}")
            return []
    
    def get_years_for_make_model(self, make: str, model: str) -> List[int]:
        """Get available years for make/model combination"""
        try:
            response = self.session.get(f"{self.base_url}/vehicle/menu/year?make={make}&model={model}")
            response.raise_for_status()
            
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            years = []
            for year_elem in root.findall('.//value'):
                try:
                    years.append(int(year_elem.text))
                except ValueError:
                    continue
            
            return sorted(years, reverse=True)
        except Exception as e:
            st.error(f"Error fetching years for {make} {model}: {str(e)}")
            return []
    
    def get_vehicle_details(self, make: str, model: str, year: int) -> List[Dict]:
        """Get detailed vehicle specifications"""
        try:
            response = self.session.get(f"{self.base_url}/vehicle?make={make}&model={model}&year={year}")
            response.raise_for_status()
            
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            vehicles = []
            for vehicle_elem in root.findall('.//vehicle'):
                vehicle_data = {}
                for child in vehicle_elem:
                    if child.text and child.text.strip():
                        vehicle_data[child.tag] = child.text.strip()
                
                # Convert numeric fields
                numeric_fields = ['city08', 'highway08', 'comb08', 'cylinders', 'displ', 'year']
                for field in numeric_fields:
                    if field in vehicle_data:
                        try:
                            vehicle_data[field] = float(vehicle_data[field])
                        except ValueError:
                            pass
                
                vehicles.append(vehicle_data)
            
            return vehicles
        except Exception as e:
            st.error(f"Error fetching vehicle details: {str(e)}")
            return []
    
    def get_electric_vehicles(self, year_range: tuple = (2018, 2025)) -> List[Dict]:
        """Get electric vehicles database"""
        try:
            all_evs = []
            
            # Common EV makes to search
            ev_makes = ['Tesla', 'Chevrolet', 'Nissan', 'BMW', 'Audi', 'Ford', 
                       'Hyundai', 'Kia', 'Volkswagen', 'Porsche', 'Mercedes-Benz',
                       'Jaguar', 'Volvo', 'Polestar', 'Lucid', 'Rivian']
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_makes = len(ev_makes)
            
            for i, make in enumerate(ev_makes):
                status_text.text(f"Fetching EV data for {make}...")
                
                # Get models for this make
                models = self.get_models_for_make(make)
                
                for model_info in models:
                    model = model_info['model']
                    
                    # Get years for this model
                    years = self.get_years_for_make_model(make, model)
                    years = [y for y in years if year_range[0] <= y <= year_range[1]]
                    
                    for year in years:
                        vehicles = self.get_vehicle_details(make, model, year)
                        
                        # Filter for electric vehicles
                        for vehicle in vehicles:
                            if self._is_electric_vehicle(vehicle):
                                vehicle['ev_range'] = self._calculate_ev_range(vehicle)
                                vehicle['efficiency'] = self._calculate_efficiency(vehicle)
                                all_evs.append(vehicle)
                        
                        # Small delay to avoid rate limiting
                        time.sleep(0.1)
                
                progress_bar.progress((i + 1) / total_makes)
            
            progress_bar.empty()
            status_text.empty()
            
            return all_evs
        
        except Exception as e:
            st.error(f"Error fetching EV database: {str(e)}")
            return []
    
    def _is_electric_vehicle(self, vehicle: Dict) -> bool:
        """Determine if vehicle is electric"""
        fuel_type = vehicle.get('fuelType', '').lower()
        fuel_type1 = vehicle.get('fuelType1', '').lower()
        
        electric_indicators = ['electricity', 'electric', 'ev']
        
        return any(indicator in fuel_type or indicator in fuel_type1 
                  for indicator in electric_indicators)
    
    def _calculate_ev_range(self, vehicle: Dict) -> Optional[float]:
        """Calculate EV range from available data"""
        try:
            # Look for range in various fields
            range_field = vehicle.get('range', vehicle.get('rangeA08', 0))
            if range_field:
                return float(range_field)
            
            # Calculate from other metrics if available
            city_mpg = vehicle.get('city08', 0)
            if city_mpg > 0:
                # For EVs, this might be MPGe, estimate range
                return city_mpg * 3.5  # Rough estimation
            
            return None
        except (ValueError, TypeError):
            return None
    
    def _calculate_efficiency(self, vehicle: Dict) -> Optional[float]:
        """Calculate efficiency metric"""
        try:
            comb_mpge = vehicle.get('comb08', 0)
            if comb_mpge > 0:
                return float(comb_mpge)
            
            city_mpge = vehicle.get('city08', 0)
            highway_mpge = vehicle.get('highway08', 0)
            
            if city_mpge > 0 and highway_mpge > 0:
                return (city_mpge + highway_mpge) / 2
            
            return None
        except (ValueError, TypeError):
            return None
