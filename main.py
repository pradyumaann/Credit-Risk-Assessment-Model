from main import CreditRiskModel

# Initialize the model
model = CreditRiskModel()

# Example data (you would replace this with real client data)
financial_data = {
    'current_assets': 1000000,
    'current_liabilities': 400000,
    'inventory': 300000,
    'total_debt': 800000,
    'total_assets': 2000000,
    'ebitda': 500000,
    'debt_service': 200000,
    'net_income': 300000,
    'ebit': 400000,
    'interest_expense': 100000
}

# Payment history data
payment_data = [
    {'status': 'PAID', 'days_late': 0},
    {'status': 'LATE', 'days_late': 15},
    {'status': 'PAID', 'days_late': 0},
    {'status': 'PAID', 'days_late': 0},
    {'status': 'LATE', 'days_late': 5},
    {'status': 'MISSED', 'days_late': None},
    {'status': 'PAID', 'days_late': 0},
]

# Market condition data
market_data = {
    'industry_growth_rate': 5.5,
    'market_share': 12.0,
    'industry_risk_score': 65,
    'economic_indicator': 3.2
}

# Qualitative assessment data
qualitative_data = {
    'management_years': 15,
    'business_model_score': 75,
    'competitive_position_score': 68,
    'compliance_score': 85
}

# Perform the assessment
assessment = model.assess_credit_risk(
    client_name="ABC Corporation",
    financial_data=financial_data,
    payment_data=payment_data,
    market_data=market_data,
    qualitative_data=qualitative_data
)

# Generate the detailed report
report = model.generate_report(assessment)

# Print the report
print(report)