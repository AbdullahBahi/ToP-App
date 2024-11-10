from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

def ensure_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def get_units_data(unit_code):
    logging.debug('Trying to read data ..')
    try:
        unit_data = pd.read_csv('units_data.csv')
    except:
        logging.debug('Failed to read data')
    project_name = unit_data.loc[unit_data['Unit Code'] == unit_code, 'Project'].values[0]
    n_bedrooms = unit_data.loc[unit_data['Unit Code'] == unit_code, 'No. of Bed Rooms'].values[0]
    finishing = unit_data.loc[unit_data['Unit Code'] == unit_code, 'Finishing Specs.'].values[0]
    gross_area = unit_data.loc[unit_data['Unit Code'] == unit_code, 'Sellable Area without Roof (Gross Area)'].values[0]
    garden_area = unit_data.loc[unit_data['Unit Code'] == unit_code, 'Garden Area'].values[0]
    penthouse_area = unit_data.loc[unit_data['Unit Code'] == unit_code, 'Penthouse Area'].values[0]
    roof_terrace_area = unit_data.loc[unit_data['Unit Code'] == unit_code, 'Penthouse Area'].values[0] + unit_data.loc[unit_data['Unit Code'] == unit_code, 'Roof Terraces Area'].values[0]
    delivery_date = unit_data.loc[unit_data['Unit Code'] == unit_code, 'Development Delivery Date'].values[0]
    base_price = unit_data.loc[unit_data['Unit Code'] == unit_code, 'Interest Free Unit Price include club'].values[0]
    maintenance_percentage = unit_data.loc[unit_data['Unit Code'] == unit_code, 'Maintenance %'].values[0]
    unit_info = {
        'Unit Code': unit_code,
        'Project Name': project_name,
        'No. of Bed Rooms': ensure_serializable(n_bedrooms),
        'Finishing Type': finishing,
        'Gross Area (sqm)': ensure_serializable(gross_area),
        'Garden Area': ensure_serializable(garden_area),
        'Penthouse Area': ensure_serializable(penthouse_area),
        'Open Terrace / Roof Area': ensure_serializable(roof_terrace_area),
        'Delivery Date': ensure_serializable(delivery_date),
        'Base Price': ensure_serializable(base_price),
        'Maintenance Fee': ensure_serializable(maintenance_percentage)
    }
    return unit_info

# Calculate increase/Decrease %
def calculate_percentage_change(base_npv, new_npv, max_discount):
    if base_npv >= new_npv:
        percentage_change = (base_npv / new_npv) - 1
    else:
        percentage_change = ((base_npv / new_npv) - 1) * (max_discount / (1 - base_npv))
    return percentage_change
    
# Calculate price with interest
def calculate_price_with_interest(base_npv, new_npv, max_discount, base_price):
    if base_npv >= new_npv:
        percentage_change = (base_npv / new_npv) - 1
    else:
        percentage_change = ((base_npv / new_npv) - 1) * (max_discount / (1 - base_npv))
    return (1 + percentage_change) * base_price

# Calculate period rate
def calculate_period_rate(interest_rate, periods_per_year):
    return (1 + interest_rate) ** (1 / periods_per_year) - 1

# Calculate installment payment amounts and dates
def calculate_installments(unit_code, tenor_years, periods_per_year, contract_date, input_pmts, interest_rate=0.294, base_dp=0.05, base_tenor_years=5, base_periods_per_year=4, max_discount=0.25):
    # Get unit info
    unit_info = get_units_data(unit_code)
    base_price = unit_info['Base Price']
    maintenance_fee_percent = unit_info['Maintenance Fee']
    ########################################################
    ## CALCULATING BASE NPV
    ########################################################
    # Calculate number of base payments
    n_base = int(base_tenor_years * base_periods_per_year)

    # Calculate the equal remaining payments after deducting the down payment and custom payments
    remaining_base_percentage = (1 - 2*base_dp) / (n_base-1)

    # Create a list with the down payment, the custom paymants, and the auto-filled payments, representing the final schedule of payments
    base_percentages = [remaining_base_percentage,]*(n_base+1)
    base_percentages[0] = base_dp
    base_percentages[1] = base_dp

    # Calculate discount rate for the period
    base_period_rate = calculate_period_rate(interest_rate, base_periods_per_year)
    
    # Calculate the Net Present Value (NPV) of base payments percentages using the period rate
    base_npv = base_dp
    for i, pmt_percent in enumerate(base_percentages[1:], start=1):
        base_npv += pmt_percent * (1 + base_period_rate) ** (-i)

    ########################################################
    ## CALCULATING NEW NPV
    ########################################################
    # Calculate number of payments
    if tenor_years is None and periods_per_year is not None:
        n = int(base_tenor_years * periods_per_year)
    elif periods_per_year is None and tenor_years is not None:
        n = int(tenor_years * base_periods_per_year)
    elif periods_per_year is None and tenor_years is None:
        n = int(base_tenor_years * base_periods_per_year)
    else:
        n = int(tenor_years * periods_per_year)

    # Extract down payment
    if not len(input_pmts):
        dp_percentage = base_dp
    else:
        dp_percentage = input_pmts[0]

    # Calculate the equal remaining payments after deducting the down payment and custom payments
    remaining_percentage = (1 - dp_percentage - sum(list(input_pmts.values())[1:])) / (n - len(list(input_pmts.values())[1:]))
    
    # Create a list with the down payment, the custom paymants, and the auto-filled payments, representing the final schedule of payments
    calculated_pmt_percentages = [0,]*(n+1)
    calculated_pmt_percentages[0] = dp_percentage
    for k, v in input_pmts.items():
        calculated_pmt_percentages[k] = v
    calculated_pmt_percentages = [p if p!=0 else remaining_percentage for p in calculated_pmt_percentages]

    # Calculate discount rate for the period
    period_rate = calculate_period_rate(interest_rate, periods_per_year)
    
    # Calculate the Net Present Value (NPV) of the calculated payments percentages using the period rate
    new_npv = dp_percentage
    for i, pmt_percent in enumerate(calculated_pmt_percentages[1:], start=1):
        new_npv += pmt_percent * (1 + period_rate) ** (-i)

    ########################################################
    ## CALCULATING PAYMENTS SCHEDULE
    ########################################################
    # Calculate the new price after interest (increase/decrease)
    price_with_interest = calculate_price_with_interest(base_npv, new_npv, max_discount, base_price)

    # Calulate the percentage of increase/decrease in the price over the interest-free price
    percentage_change = calculate_percentage_change(base_npv, new_npv, max_discount)

    # Calculate a list of payments amounts
    pmt_amounts = [percent * price_with_interest for percent in calculated_pmt_percentages]

    # Calculate payments dates
    if contract_date is None:
        pmt_dates = [(datetime.today() + timedelta(days=int(365 / periods_per_year) * i)).strftime("%Y-%m-%d") for i in range(n+1)]
    else:
        contract_date = datetime.strptime(contract_date, "%Y-%m-%d")
        pmt_dates = [(contract_date + timedelta(days=int(365 / periods_per_year) * i)).strftime("%Y-%m-%d") for i in range(n+1)]
        
    # calculate cumulative payments percentage
    cumulative_pmt_percent = [sum(calculated_pmt_percentages[:i+1]) for i in range(n+1)]

    # Create a list of payment types
    pmt_type = ["PMT "+str(i+1) for i in range(n)]
    pmt_type = ["DP"] + pmt_type

    # Pack the data into a dictionary
    payemnts_schedule = {
        "Unit Info": unit_info,
        "Discount Rate [Per Period]": period_rate,
        "Base NPV": base_npv,
        "New NPV": new_npv,
        "Price With Interest": price_with_interest,
        "Increase/Decrease Percentage": percentage_change,
        "Maintenance Fee": maintenance_fee_percent * price_with_interest,
        "PMT Type":pmt_type,
        "Payment Date":pmt_dates,
        "Payment Percentage":calculated_pmt_percentages,
        "Payment Amount":pmt_amounts,
        "Cumulative Percentage":cumulative_pmt_percent
    }
    return payemnts_schedule

@app.route('/calculate_installments', methods=['POST'])
def calculate_installments_api():
    try:
        # Extract data from request
        data = request.json
        
        # Required fields
        unit_code = data['unit_code']
        tenor_years = data['tenor_years']
        periods_per_year = data['periods_per_year']
        input_pmts = data['input_pmts']
        contract_date = data['contract_date']
        input_pmts = {int(k):v for k, v in input_pmts.items()}
        # Call the calculate_installments function
        payment_schedule = calculate_installments(
            unit_code=unit_code,
            tenor_years=tenor_years,
            periods_per_year=periods_per_year,
            contract_date=contract_date,
            input_pmts=input_pmts
        )
        return jsonify(payment_schedule), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/calculate_installments', methods=['GET'])
def get_example():
    return jsonify({
        'message': 'Use POST request with JSON payload to calculate installments.'
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
