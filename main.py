import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

class CreditRiskModel:
    def __init__(self):
        # Weights for different components
        self.weights = {
            'financial_score': 0.35,
            'behavioral_score': 0.25,
            'market_score': 0.20,
            'qualitative_score': 0.20
        }
        
        # Thresholds for risk categorization
        self.risk_thresholds = {
            'LOW': 80,
            'MEDIUM': 60,
            'HIGH': 40
        }
        
        # Industry benchmarks
        self.industry_benchmarks = {
            'current_ratio': 2.0,
            'quick_ratio': 1.0,
            'debt_ratio': 0.5,
            'debt_service_coverage': 1.25,
            'return_on_assets': 0.05
        }

    def calculate_financial_ratios(self, financial_data: Dict) -> Dict:
        """Calculate key financial ratios from financial statements."""
        try:
            ratios = {
                'current_ratio': financial_data['current_assets'] / financial_data['current_liabilities'],
                'quick_ratio': (financial_data['current_assets'] - financial_data['inventory']) / 
                              financial_data['current_liabilities'],
                'debt_ratio': financial_data['total_debt'] / financial_data['total_assets'],
                'debt_service_coverage': financial_data['ebitda'] / financial_data['debt_service'],
                'return_on_assets': financial_data['net_income'] / financial_data['total_assets'],
                'interest_coverage': financial_data['ebit'] / financial_data['interest_expense']
            }
            return ratios
        except ZeroDivisionError:
            raise ValueError("Invalid financial data: Division by zero encountered")
        except KeyError as e:
            raise KeyError(f"Missing required financial data: {str(e)}")

    def score_financial_ratios(self, ratios: Dict) -> Tuple[float, Dict]:
        """Score financial ratios against industry benchmarks."""
        scores = {}
        for ratio, value in ratios.items():
            if ratio in self.industry_benchmarks:
                benchmark = self.industry_benchmarks[ratio]
                # Score from 0 to 100 based on ratio performance
                if ratio in ['debt_ratio']:  # Lower is better
                    scores[ratio] = max(0, min(100, (1 - value/benchmark) * 100))
                else:  # Higher is better
                    scores[ratio] = max(0, min(100, (value/benchmark) * 100))
        
        return np.mean(list(scores.values())), scores

    def analyze_payment_history(self, payment_data: List[Dict]) -> Tuple[float, Dict]:
        """Analyze historical payment behavior."""
        if not payment_data:
            raise ValueError("Payment history data is required")
            
        analysis = {
            'late_payments': 0,
            'average_days_late': 0,
            'missed_payments': 0,
            'total_payments': len(payment_data)
        }
        
        days_late = []
        for payment in payment_data:
            if payment['status'] == 'LATE':
                analysis['late_payments'] += 1
                days_late.append(payment['days_late'])
            elif payment['status'] == 'MISSED':
                analysis['missed_payments'] += 1
                
        analysis['average_days_late'] = np.mean(days_late) if days_late else 0
        
        # Calculate behavioral score (0-100)
        on_time_ratio = (analysis['total_payments'] - analysis['late_payments'] - 
                        analysis['missed_payments']) / analysis['total_payments']
        behavioral_score = on_time_ratio * 100 - (analysis['average_days_late'] * 0.5)
        behavioral_score = max(0, min(100, behavioral_score))
        
        return behavioral_score, analysis

