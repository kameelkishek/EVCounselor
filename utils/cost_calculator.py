import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import streamlit as st

class CostCalculator:
    """Calculate cost-benefit analysis for EV recommendations"""
    
    def __init__(self):
        # Default values for calculations
        self.default_gas_price = 3.50  # per gallon
        self.default_electricity_cost = 0.13  # per kWh
        self.ev_efficiency_kwh_per_mile = 0.3  # average kWh per mile for EVs
        self.maintenance_savings_per_mile = 0.05  # EV maintenance savings
        self.federal_tax_credit = 7500  # Federal EV tax credit
        
        # Price estimates for EV categories
        self.ev_price_estimates = {
            1: {'min': 25000, 'max': 35000, 'avg': 30000},  # Budget
            2: {'min': 35000, 'max': 55000, 'avg': 45000},  # Mid-range
            3: {'min': 55000, 'max': 100000, 'avg': 75000}  # Luxury
        }
    
    def calculate_cost_analysis(self, original_vehicle: Dict, ev_recommendation: Dict, 
                              analysis_years: int = 5) -> Dict:
        """Calculate comprehensive cost analysis"""
        try:
            # Extract vehicle data
            annual_mileage = original_vehicle.get('annual_mileage', 
                                                original_vehicle.get('daily_mileage', 40) * 365)
            current_mpg = original_vehicle.get('current_mpg', 25)
            
            # EV data
            ev_efficiency = ev_recommendation.get('ev_efficiency', 100)  # MPGe
            ev_price_category = ev_recommendation.get('ev_price_category', 2)
            
            # Calculate costs
            gas_costs = self._calculate_gas_costs(annual_mileage, current_mpg, analysis_years)
            electricity_costs = self._calculate_electricity_costs(annual_mileage, ev_efficiency, analysis_years)
            maintenance_costs = self._calculate_maintenance_costs(annual_mileage, analysis_years)
            vehicle_costs = self._calculate_vehicle_costs(ev_price_category)
            
            # Calculate savings
            fuel_savings = gas_costs['total'] - electricity_costs['total']
            maintenance_savings = maintenance_costs['gas_vehicle'] - maintenance_costs['ev']
            total_operational_savings = fuel_savings + maintenance_savings
            
            # Net cost analysis
            net_cost = vehicle_costs['net_price'] - total_operational_savings
            payback_period = self._calculate_payback_period(
                vehicle_costs['net_price'], 
                (fuel_savings + maintenance_savings) / analysis_years
            )
            
            return {
                'analysis_years': analysis_years,
                'annual_mileage': annual_mileage,
                'gas_costs': gas_costs,
                'electricity_costs': electricity_costs,
                'maintenance_costs': maintenance_costs,
                'vehicle_costs': vehicle_costs,
                'fuel_savings': fuel_savings,
                'maintenance_savings': maintenance_savings,
                'total_operational_savings': total_operational_savings,
                'net_cost': net_cost,
                'payback_period_years': payback_period,
                'cost_per_mile_gas': gas_costs['total'] / (annual_mileage * analysis_years),
                'cost_per_mile_ev': electricity_costs['total'] / (annual_mileage * analysis_years),
                'roi_percentage': (total_operational_savings / vehicle_costs['net_price']) * 100 if vehicle_costs['net_price'] > 0 else 0
            }
        
        except Exception as e:
            st.error(f"Error calculating cost analysis: {str(e)}")
            return {}
    
    def _calculate_gas_costs(self, annual_mileage: float, mpg: float, years: int) -> Dict:
        """Calculate gasoline costs over analysis period"""
        annual_gallons = annual_mileage / mpg
        annual_cost = annual_gallons * self.default_gas_price
        total_cost = annual_cost * years
        
        return {
            'annual_gallons': annual_gallons,
            'annual_cost': annual_cost,
            'total': total_cost,
            'per_mile': annual_cost / annual_mileage
        }
    
    def _calculate_electricity_costs(self, annual_mileage: float, mpge: float, years: int) -> Dict:
        """Calculate electricity costs for EV"""
        # Convert MPGe to kWh per mile (rough conversion)
        kwh_per_mile = 33.7 / mpge  # 33.7 kWh equivalent to 1 gallon
        
        annual_kwh = annual_mileage * kwh_per_mile
        annual_cost = annual_kwh * self.default_electricity_cost
        total_cost = annual_cost * years
        
        return {
            'annual_kwh': annual_kwh,
            'annual_cost': annual_cost,
            'total': total_cost,
            'per_mile': annual_cost / annual_mileage,
            'kwh_per_mile': kwh_per_mile
        }
    
    def _calculate_maintenance_costs(self, annual_mileage: float, years: int) -> Dict:
        """Calculate maintenance costs comparison"""
        # Maintenance cost per mile estimates
        gas_maintenance_per_mile = 0.12
        ev_maintenance_per_mile = 0.07
        
        total_miles = annual_mileage * years
        
        gas_vehicle_maintenance = total_miles * gas_maintenance_per_mile
        ev_maintenance = total_miles * ev_maintenance_per_mile
        
        return {
            'gas_vehicle': gas_vehicle_maintenance,
            'ev': ev_maintenance,
            'savings': gas_vehicle_maintenance - ev_maintenance,
            'savings_per_mile': gas_maintenance_per_mile - ev_maintenance_per_mile
        }
    
    def _calculate_vehicle_costs(self, price_category: int) -> Dict:
        """Calculate vehicle purchase costs"""
        price_info = self.ev_price_estimates.get(price_category, self.ev_price_estimates[2])
        
        estimated_price = price_info['avg']
        net_price = max(0, estimated_price - self.federal_tax_credit)
        
        return {
            'estimated_price': estimated_price,
            'federal_tax_credit': self.federal_tax_credit,
            'net_price': net_price,
            'price_range': f"${price_info['min']:,} - ${price_info['max']:,}"
        }
    
    def _calculate_payback_period(self, initial_cost: float, annual_savings: float) -> float:
        """Calculate payback period in years"""
        if annual_savings <= 0:
            return float('inf')
        
        return initial_cost / annual_savings
    
    def calculate_fleet_cost_summary(self, fleet_recommendations: pd.DataFrame) -> Dict:
        """Calculate cost summary for entire fleet"""
        try:
            total_vehicles = len(fleet_recommendations)
            if total_vehicles == 0:
                return {}
            
            # Group by original vehicle to avoid duplicates
            unique_vehicles = fleet_recommendations.drop_duplicates(subset=['vehicle_id'])
            
            total_fuel_savings = 0
            total_maintenance_savings = 0
            total_vehicle_costs = 0
            total_net_savings = 0
            
            for _, row in unique_vehicles.iterrows():
                # Get top recommendation for each vehicle
                vehicle_recs = fleet_recommendations[
                    fleet_recommendations['vehicle_id'] == row['vehicle_id']
                ].sort_values('similarity_score', ascending=False)
                
                if not vehicle_recs.empty:
                    top_rec = vehicle_recs.iloc[0]
                    
                    # Calculate cost analysis for this vehicle
                    cost_analysis = self.calculate_cost_analysis(
                        row.to_dict(), 
                        top_rec.to_dict()
                    )
                    
                    if cost_analysis:
                        total_fuel_savings += cost_analysis.get('fuel_savings', 0)
                        total_maintenance_savings += cost_analysis.get('maintenance_savings', 0)
                        total_vehicle_costs += cost_analysis.get('vehicle_costs', {}).get('net_price', 0)
                        total_net_savings += cost_analysis.get('total_operational_savings', 0) - cost_analysis.get('vehicle_costs', {}).get('net_price', 0)
            
            avg_payback = (total_vehicle_costs / (total_fuel_savings + total_maintenance_savings) * 12) if (total_fuel_savings + total_maintenance_savings) > 0 else 0
            
            return {
                'total_vehicles': total_vehicles,
                'total_fuel_savings_5yr': total_fuel_savings,
                'total_maintenance_savings_5yr': total_maintenance_savings,
                'total_operational_savings_5yr': total_fuel_savings + total_maintenance_savings,
                'total_vehicle_investment': total_vehicle_costs,
                'net_fleet_savings_5yr': total_net_savings,
                'average_payback_months': avg_payback,
                'roi_percentage': (total_fuel_savings + total_maintenance_savings) / total_vehicle_costs * 100 if total_vehicle_costs > 0 else 0
            }
        
        except Exception as e:
            st.error(f"Error calculating fleet cost summary: {str(e)}")
            return {}
    
    def create_cost_comparison_chart_data(self, cost_analysis: Dict) -> Dict:
        """Prepare data for cost comparison charts"""
        try:
            years = list(range(1, cost_analysis.get('analysis_years', 5) + 1))
            
            # Cumulative costs
            annual_gas_cost = cost_analysis.get('gas_costs', {}).get('annual_cost', 0)
            annual_electricity_cost = cost_analysis.get('electricity_costs', {}).get('annual_cost', 0)
            annual_maintenance_savings = cost_analysis.get('maintenance_savings', 0) / cost_analysis.get('analysis_years', 5)
            
            gas_cumulative = [annual_gas_cost * year for year in years]
            ev_cumulative = [annual_electricity_cost * year for year in years]
            savings_cumulative = [(annual_gas_cost - annual_electricity_cost + annual_maintenance_savings) * year for year in years]
            
            return {
                'years': years,
                'gas_costs': gas_cumulative,
                'ev_costs': ev_cumulative,
                'cumulative_savings': savings_cumulative,
                'break_even_point': cost_analysis.get('payback_period_years', 0)
            }
        
        except Exception as e:
            st.error(f"Error preparing chart data: {str(e)}")
            return {}
