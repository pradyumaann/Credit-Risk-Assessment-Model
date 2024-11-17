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

    def assess_market_conditions(self, market_data: Dict) -> Tuple[float, Dict]:
        """Assess market and industry conditions."""
        market_analysis = {
            'industry_growth': self._normalize_score(market_data['industry_growth_rate'], -5, 15),
            'market_position': self._normalize_score(market_data['market_share'], 0, 30),
            'industry_risk': self._normalize_score(market_data['industry_risk_score'], 100, 0),
            'economic_conditions': self._normalize_score(market_data['economic_indicator'], -10, 10)
        }
        
        market_score = np.mean(list(market_analysis.values()))
        return market_score, market_analysis

    def evaluate_qualitative_factors(self, qualitative_data: Dict) -> Tuple[float, Dict]:
        """Evaluate qualitative risk factors."""
        qualitative_analysis = {
            'management_experience': self._normalize_score(qualitative_data['management_years'], 0, 20),
            'business_model': qualitative_data['business_model_score'],
            'competitive_position': qualitative_data['competitive_position_score'],
            'regulatory_compliance': qualitative_data['compliance_score']
        }
        
        qualitative_score = np.mean(list(qualitative_analysis.values()))
        return qualitative_score, qualitative_analysis

    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to a 0-100 scale."""
        return max(0, min(100, ((value - min_val) / (max_val - min_val)) * 100))

    def calculate_probability_of_default(self, total_score: float) -> float:
        """Calculate probability of default using a logistic function."""
        # Simple logistic function transformation
        return 1 / (1 + np.exp(-0.1 * (100 - total_score)))

    def generate_risk_rating(self, score: float) -> str:
        """Generate risk rating based on score."""
        if score >= self.risk_thresholds['LOW']:
            return 'LOW RISK'
        elif score >= self.risk_thresholds['MEDIUM']:
            return 'MEDIUM RISK'
        elif score >= self.risk_thresholds['HIGH']:
            return 'HIGH RISK'
        else:
            return 'VERY HIGH RISK'

    def assess_credit_risk(self, 
                          client_name: str,
                          financial_data: Dict,
                          payment_data: List[Dict],
                          market_data: Dict,
                          qualitative_data: Dict) -> Dict:
        """Perform comprehensive credit risk assessment."""
        
        # Calculate all component scores
        financial_ratios = self.calculate_financial_ratios(financial_data)
        financial_score, financial_analysis = self.score_financial_ratios(financial_ratios)
        
        behavioral_score, behavioral_analysis = self.analyze_payment_history(payment_data)
        
        market_score, market_analysis = self.assess_market_conditions(market_data)
        
        qualitative_score, qualitative_analysis = self.evaluate_qualitative_factors(qualitative_data)
        
        # Calculate weighted total score
        total_score = (
            financial_score * self.weights['financial_score'] +
            behavioral_score * self.weights['behavioral_score'] +
            market_score * self.weights['market_score'] +
            qualitative_score * self.weights['qualitative_score']
        )
        
        # Calculate PD and risk rating
        pd = self.calculate_probability_of_default(total_score)
        risk_rating = self.generate_risk_rating(total_score)
        
        # Compile comprehensive assessment
        assessment = {
            'client_name': client_name,
            'assessment_date': datetime.now().strftime('%Y-%m-%d'),
            'total_score': round(total_score, 2),
            'risk_rating': risk_rating,
            'probability_of_default': round(pd * 100, 2),
            'component_scores': {
                'financial_score': round(financial_score, 2),
                'behavioral_score': round(behavioral_score, 2),
                'market_score': round(market_score, 2),
                'qualitative_score': round(qualitative_score, 2)
            },
            'detailed_analysis': {
                'financial_analysis': financial_analysis,
                'financial_ratios': financial_ratios,
                'behavioral_analysis': behavioral_analysis,
                'market_analysis': market_analysis,
                'qualitative_analysis': qualitative_analysis
            }
        }
        
        return assessment

    def generate_report(self, assessment: Dict) -> str:
        """Generate a detailed credit risk assessment report."""
        report = f"""
CREDIT RISK ASSESSMENT REPORT
============================
Client: {assessment['client_name']}
Date: {assessment['assessment_date']}

EXECUTIVE SUMMARY
----------------
Overall Risk Score: {assessment['total_score']}/100
Risk Rating: {assessment['risk_rating']}
Probability of Default: {assessment['probability_of_default']}%

COMPONENT SCORES
---------------
Financial Score: {assessment['component_scores']['financial_score']}/100
Behavioral Score: {assessment['component_scores']['behavioral_score']}/100
Market Score: {assessment['component_scores']['market_score']}/100
Qualitative Score: {assessment['component_scores']['qualitative_score']}/100

DETAILED ANALYSIS
----------------
1. Financial Analysis
--------------------
"""
        
        # Add financial ratios analysis
        report += "Financial Ratios:\n"
        for ratio, value in assessment['detailed_analysis']['financial_ratios'].items():
            benchmark = self.industry_benchmarks.get(ratio, "N/A")
            report += f"- {ratio.replace('_', ' ').title()}: {value:.2f} (Benchmark: {benchmark})\n"
        
        # Add behavioral analysis
        report += "\n2. Payment Behavior Analysis\n"
        report += "-------------------------\n"
        behavior = assessment['detailed_analysis']['behavioral_analysis']
        report += f"Total Payments Analyzed: {behavior['total_payments']}\n"
        report += f"Late Payments: {behavior['late_payments']}\n"
        report += f"Average Days Late: {behavior['average_days_late']:.1f}\n"
        report += f"Missed Payments: {behavior['missed_payments']}\n"
        
        # Add market analysis
        report += "\n3. Market Condition Analysis\n"
        report += "-------------------------\n"
        for factor, score in assessment['detailed_analysis']['market_analysis'].items():
            report += f"{factor.replace('_', ' ').title()}: {score:.1f}/100\n"
        
        # Add qualitative analysis
        report += "\n4. Qualitative Factors Analysis\n"
        report += "-----------------------------\n"
        for factor, score in assessment['detailed_analysis']['qualitative_analysis'].items():
            report += f"{factor.replace('_', ' ').title()}: {score:.1f}/100\n"
        
        # Add risk assessment summary
        report += f"""
RISK ASSESSMENT SUMMARY
----------------------
The client presents a {assessment['risk_rating'].lower()} profile with a {assessment['probability_of_default']}% 
probability of default. This assessment is based on a comprehensive analysis of financial, 
behavioral, market, and qualitative factors.

Key Strengths:
- {self._identify_strengths(assessment)}

Key Concerns:
- {self._identify_concerns(assessment)}

RECOMMENDATION
-------------
{self._generate_recommendation(assessment)}
"""
        
        return report

    def _identify_strengths(self, assessment: Dict) -> str:
        """Identify key strengths from the assessment."""
        strengths = []
        
        # Check financial ratios
        financial_ratios = assessment['detailed_analysis']['financial_ratios']
        for ratio, value in financial_ratios.items():
            benchmark = self.industry_benchmarks.get(ratio)
            if benchmark and ratio != 'debt_ratio':
                if value > benchmark * 1.2:  # 20% better than benchmark
                    strengths.append(f"Strong {ratio.replace('_', ' ')} ({value:.2f} vs benchmark {benchmark})")
            elif ratio == 'debt_ratio' and value < benchmark * 0.8:  # 20% better than benchmark
                strengths.append(f"Low {ratio.replace('_', ' ')} ({value:.2f} vs benchmark {benchmark})")
        
        # Check behavioral score
        if assessment['component_scores']['behavioral_score'] > 80:
            strengths.append("Excellent payment history")
        
        return '\n- '.join(strengths) if strengths else "No significant strengths identified"

    def _identify_concerns(self, assessment: Dict) -> str:
        """Identify key concerns from the assessment."""
        concerns = []
        
        # Check financial ratios
        financial_ratios = assessment['detailed_analysis']['financial_ratios']
        for ratio, value in financial_ratios.items():
            benchmark = self.industry_benchmarks.get(ratio)
            if benchmark and ratio != 'debt_ratio':
                if value < benchmark * 0.8:  # 20% worse than benchmark
                    concerns.append(f"Weak {ratio.replace('_', ' ')} ({value:.2f} vs benchmark {benchmark})")
            elif ratio == 'debt_ratio' and value > benchmark * 1.2:  # 20% worse than benchmark
                concerns.append(f"High {ratio.replace('_', ' ')} ({value:.2f} vs benchmark {benchmark})")
        
        # Check behavioral score
        if assessment['component_scores']['behavioral_score'] < 60:
            concerns.append("Poor payment history")
        
        return '\n- '.join(concerns) if concerns else "No significant concerns identified"

    def _generate_recommendation(self, assessment: Dict) -> str:
        """Generate a recommendation based on the assessment."""
        score = assessment['total_score']
        pd = assessment['probability_of_default']
        
        if score >= self.risk_thresholds['LOW']:
            return "Recommended for approval with standard terms and conditions."
        elif score >= self.risk_thresholds['MEDIUM']:
            return "Recommended for approval with enhanced monitoring and possible additional collateral requirements."
        elif score >= self.risk_thresholds['HIGH']:
            return "Recommended for approval only with substantial collateral and restrictive covenants."
        else:
            return "Not recommended for approval under current conditions."