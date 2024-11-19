import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import json

PERIODS_PER_YEAR = {
    "monthly":12,
    "quarterly":4,
    "semi-annually":2,
    "annually":1
}

def get_units_data(unit_code):

    file_path = os.path.join(os.path.dirname(__file__), 'units_data.csv')
    unit_data = pd.read_csv(file_path)

    project_name = unit_data.loc[unit_data['Unit Code'] == unit_code, 'Project'].values[0]

    #Extract project data
    file_path = os.path.join(os.path.dirname(__file__), f'policies/{project_name}.json')
    with open(file_path, "r") as f:
        project_policy = json.load(f)
    
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
        'Maintenance Fee': ensure_serializable(maintenance_percentage),
        'Project Policy':project_policy
    }
    return unit_info

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
        
# Calculate increase/Decrease %
def calculate_percentage_change(base_npv, new_npv, max_discount):
    if base_npv >= new_npv:
        percentage_change = (base_npv / new_npv) - 1
    else:
        percentage_change = ((base_npv / new_npv) - 1) * (max_discount / (1 - base_npv))
    return percentage_change
    
# Calculate price with interest
def calculate_price_with_interest(base_npv, new_npv, max_discount, base_price):
    print("base_npv", base_npv)
    print("new_npv", new_npv)
    print("max_discount", max_discount)
    print("base_base_pricenpv", base_price)
    if base_npv >= new_npv:
        percentage_change = (base_npv / new_npv) - 1
    else:
        percentage_change = ((base_npv / new_npv) - 1) * (max_discount / (1 - base_npv))
    print("percentage_change", percentage_change)
    print("price_with_interest", (1 + percentage_change) * base_price)
    return (1 + percentage_change) * base_price

# Calculate period rate
def calculate_period_rate(interest_rate, periods_per_year):
    return (1 + interest_rate) ** (1 / periods_per_year) - 1

def caclulate_years_till_delivery(contract_date, delivery_date):
    delivery_date = datetime.strptime(delivery_date, "%m/%d/%Y")
    n_years = (delivery_date-contract_date).days / 365
    return n_years

# Calculate gas payments 
def calculate_gas_payments(policy, tenor_years, periods_per_year, contract_date, delivery_date):
    num_pmts = policy['num_pmts']
    scheduling = policy['scheduling']
    fees = policy['fees']
    fees = {float(k):v for k, v in fees.items()}
    
    # Select gas fee
    years_till_delivery = caclulate_years_till_delivery(contract_date, delivery_date)
    diffs = {abs(years_till_delivery-k):v for k, v in fees.items()}
    gas_fee = diffs[min(diffs.keys())]

    # Schedule gas fee
    delivery_payment_index = round(years_till_delivery * periods_per_year)
    
    if years_till_delivery > tenor_years:
        n = delivery_payment_index
    else:
        n = tenor_years * periods_per_year
    
    n = tenor_years * periods_per_year
    gas_payments = [0,]*(n+1)

    if scheduling == "at_delivery":
        gas_payments[delivery_payment_index] = gas_fee
    elif scheduling == "before_delivery":
        gas_pmt = gas_fee / num_pmts 
        for i in range(num_pmts):
            gas_payments[delivery_payment_index+i-num_pmts] = gas_pmt

    return gas_payments

# Calculate maintenance payments 
def calculate_maintenance_payments(policy, maintenance_fee, tenor_years, periods_per_year, contract_date, delivery_date):
    num_pmts = policy['num_pmts']
    scheduling = policy['scheduling']
    
    # Schedule maintenance fee
    years_till_delivery = caclulate_years_till_delivery(contract_date, delivery_date)
    delivery_payment_index = round(years_till_delivery * periods_per_year)
    
    if years_till_delivery > tenor_years:
        n = delivery_payment_index
    else:
        n = tenor_years * periods_per_year
    
    maintenance_payments = [0,]*(n+1)

    if scheduling == "at_delivery":
        maintenance_payments[delivery_payment_index] = maintenance_fee
    elif scheduling == "before_delivery":
        maintenance_pmt = maintenance_fee / num_pmts 
        for i in range(num_pmts):
            maintenance_payments[delivery_payment_index+i-num_pmts] = maintenance_pmt

    return maintenance_payments

# Apply constraints
def apply_constraints(pmt_percentages, tenor_years, periods_per_year, input_pmts, constraints, contract_date, delivery_date):
    
    # Apply all constraints in the project policy
    ## Handle minimum down payment constraint
    if pmt_percentages[0] < constraints['dp_min']:
        pmt_percentages[0] = constraints['dp_min']
    
    ## Handle minimum down payment plus first payment constraint
    if pmt_percentages[0] + pmt_percentages[1] < constraints['dp_plus_first_pmt']:
        pmt_percentages[1] = constraints['dp_plus_first_pmt']-pmt_percentages[0]

    # Calculate the equal remaining payments after deducting the down payment, first payment, and custom payments
    n = int(tenor_years * periods_per_year)
    if len(list(input_pmts.values())[1:]) == 0:
        remaining_percentage = (1 - pmt_percentages[0] - pmt_percentages[1] - sum(list(input_pmts.values())[2:])) / (n - 1)
    else:
        remaining_percentage = (1 - pmt_percentages[0] - pmt_percentages[1] - sum(list(input_pmts.values())[2:])) / (n - len(list(input_pmts.values())[1:]))
    
    for k, v in input_pmts.items():
        if k==0 or k==1:
            continue
        pmt_percentages[k] = v
    pmt_percentages = [p if p!=0 else remaining_percentage for p in pmt_percentages]
    
    ## Handle first year minimum constraint
    first_year_payments = pmt_percentages[:periods_per_year+1]
    
    if sum(first_year_payments) < constraints['first_year_min']:
        pmt_percentages[periods_per_year] = constraints['first_year_min'] - sum(first_year_payments[:-1])

    if sum(pmt_percentages) > 1:
        sum_after_first_year = sum(pmt_percentages[periods_per_year+1:])
        
        total_custom_payments_after_first_year = 0
        num_custom_payments_after_first_year = 0
        
        for k in input_pmts.keys():
            if k <= periods_per_year:
                continue
            total_custom_payments_after_first_year += pmt_percentages[k]
            num_custom_payments_after_first_year += 1
        
        excess = sum(pmt_percentages) - 1
        
        new_remaining_percentage = (sum_after_first_year-total_custom_payments_after_first_year-excess) / (len(pmt_percentages[periods_per_year+1:]) - num_custom_payments_after_first_year)
        
        for i, pmt in enumerate(pmt_percentages[periods_per_year+1:]):
            if pmt == remaining_percentage:
                pmt_percentages[periods_per_year+1+i] = new_remaining_percentage
        
        remaining_percentage = new_remaining_percentage
    
    ## Handle cumulative minimum constraint 
    for year in range(tenor_years):
        # year_payments = pmt_percentages[(year*periods_per_year)+1:((year+1)*periods_per_year)+1]
        cummulative_payments = pmt_percentages[:((year+1)*periods_per_year)+1]
        
        if sum(cummulative_payments) < (year * constraints['annual_min']) + constraints['first_year_min']:
            pmt_percentages[(year+1)*periods_per_year] = (year * constraints['annual_min']) + constraints['first_year_min'] - sum(cummulative_payments[:-1])

    ## Handle cash till delivery constraint
    years_till_delivery = caclulate_years_till_delivery(contract_date, delivery_date)
    
    ctd_mins = constraints['ctd_min']
    ctd_mins = {float(k):v for k, v in ctd_mins.items()}
    
    diffs = {abs(years_till_delivery-k):v for k, v in ctd_mins.items()}
    
    ctd = diffs[min(diffs.keys())]
    
    delivery_payment_index = round(years_till_delivery * periods_per_year)
    
    payments_till_delivery = pmt_percentages[:delivery_payment_index+1]
    
    if sum(payments_till_delivery) < ctd:
        pmt_percentages[delivery_payment_index] = ctd - sum(payments_till_delivery[:-1])

    if sum(pmt_percentages) > 1:
        sum_after_delivery = sum(pmt_percentages[delivery_payment_index+1:])
        total_custom_payments_after_delivery = 0
        num_custom_payments_after_delivery = 0
        for k in input_pmts.keys():
            if k <= delivery_payment_index:
                continue
            total_custom_payments_after_delivery += pmt_percentages[k]
            num_custom_payments_after_delivery += 1
        excess = sum(pmt_percentages) - 1
        
        new_remaining_percentage = (sum_after_delivery-total_custom_payments_after_delivery-excess) / (len(pmt_percentages[delivery_payment_index+1:]) - num_custom_payments_after_delivery)
        
        for i, pmt in enumerate(pmt_percentages[delivery_payment_index+1:]):
            if pmt == remaining_percentage:
                pmt_percentages[delivery_payment_index+1+i] = new_remaining_percentage
        
        remaining_percentage = new_remaining_percentage
        
    return pmt_percentages

# Calculate installment payment amounts and dates
def calculate_installments(unit_info, tenor_years, payment_frequency, contract_date, input_pmts):
    # Get unit info
    base_price = unit_info['Base Price']
    project_policy = unit_info['Project Policy']
    interest_rate = project_policy['interest_rate']
    base_dp = project_policy['base_dp']
    base_tenor_years = project_policy['base_tenor_years']
    base_payment_frequency = project_policy['base_payment_frequency']
    max_discount = project_policy['constraints']['max_discount']
    maintenance_fee_percent = unit_info['Maintenance Fee']

    del unit_info['Project Policy']
    
    if contract_date is None:
        contract_date = datetime.today()
    else:
        contract_date = datetime.strptime(contract_date, "%Y-%m-%d")
        
    max_tenor_years = int((1-project_policy['constraints']['first_year_min'])/project_policy['constraints']['annual_min']) + 1 ## Force maximum tenor years based on project constraints
    if tenor_years > max_tenor_years:
        tenor_years = max_tenor_years
    elif tenor_years == 0:
        tenor_years = base_tenor_years

    if project_policy['use_static_base_npv']:
        base_npvs = project_policy['base_npv']
        base_npvs = {float(k):v for k, v in base_npvs.items()}
        diffs = {abs(tenor_years-k):v for k, v in base_npvs.items()}
        base_npv = diffs[min(diffs.keys())]
    else:
        ########################################################
        ## CALCULATING BASE NPV
        ########################################################
        # Calculate number of base payments
        base_periods_per_year = PERIODS_PER_YEAR[base_payment_frequency.lower()]
        n_base = int(base_tenor_years * base_periods_per_year)
        
        # Create a list with the down payment, the custom paymants, and the auto-filled payments, representing the final schedule of payments
        base_percentages = [0,]*(n_base+1)
        base_percentages[0] = base_dp
        
        # Apply constraints on the calculated list of payment percentages
        base_percentages = apply_constraints(base_percentages, base_tenor_years, base_periods_per_year, {}, project_policy['constraints'], contract_date, unit_info['Delivery Date'])
        
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
    periods_per_year = PERIODS_PER_YEAR[payment_frequency.lower()]
    n = int(tenor_years * periods_per_year)

    # Extract down payment
    if len(input_pmts) == 0:
        dp_percentage = base_dp
    else:
        dp_percentage = input_pmts[0]

    # Create a list with the down payment, the custom paymants, and the auto-filled payments, representing the final schedule of payments
    calculated_pmt_percentages = [0,]*(n+1)
    calculated_pmt_percentages[0] = dp_percentage

    # Apply constraints on the calculated list of payment percentages
    calculated_pmt_percentages = apply_constraints(calculated_pmt_percentages, tenor_years, periods_per_year, input_pmts, project_policy['constraints'], contract_date, unit_info['Delivery Date'])
    
    # Calculate discount rate for the period
    period_rate = calculate_period_rate(interest_rate, periods_per_year)
    # Calculate the Net Present Value (NPV) of the calculated payments percentages using the period rate
    new_npv = calculated_pmt_percentages[0]
    for i, pmt_percent in enumerate(calculated_pmt_percentages[1:], start=1):
        new_npv += pmt_percent * (1 + period_rate) ** (-i)
    
    ########################################################
    ## CALCULATING PAYMENTS SCHEDULE
    ########################################################
    # Calculate the new price after interest (increase/decrease)
    price_with_interest = calculate_price_with_interest(base_npv, new_npv, max_discount, base_price)
    
    # Calulate the percentage of increase/decrease in the price over the interest-free price
    percentage_change = calculate_percentage_change(base_npv, new_npv, max_discount)
    print(1)
    # Calculate a list of payments amounts
    pmt_amounts = [percent * price_with_interest for percent in calculated_pmt_percentages]
    print(2)
    # Calculate gas payments 
    if project_policy['gas_policy']['is_applied']:
        gas_payments = calculate_gas_payments(project_policy['gas_policy'], tenor_years, periods_per_year, contract_date, unit_info['Delivery Date'])
    else:
        gas_payments = gas_payments = [0,]*(n+1)
    print(3)
    # Calculate maintenance payments 
    if project_policy['maintenance_policy']['is_applied']:
        maintenance_payments = calculate_maintenance_payments(project_policy['maintenance_policy'], maintenance_fee_percent * price_with_interest, tenor_years, periods_per_year, contract_date, unit_info['Delivery Date'])
    else:
       maintenance_payments = gas_payments = [0,]*(n+1)
    print(4)
    # Calculate payments dates
    pmt_dates = [(contract_date + timedelta(days=int(365 / periods_per_year) * i)).strftime("%Y-%m-%d") for i in range(n+1)]
    print(5)
    # calculate cumulative payments percentage
    cumulative_pmt_percent = [sum(calculated_pmt_percentages[:i+1]) for i in range(n+1)]
    print(6)
    # Create a list of payment types
    pmt_type = ["PMT "+str(i+1) for i in range(n)]
    pmt_type = ["DP"] + pmt_type
    print(7)
    # Pack the data into a dictionary
    if len(gas_payments) > len(pmt_type) or len(maintenance_payments) > len(pmt_type):
        pmt_type += ["PMT "+str(i+1+len(pmt_type)) for i in range(max(len(gas_payments)-len(pmt_type), len(maintenance_payments)-len(pmt_type)))] 
        print(71, len(pmt_type))
        pmt_dates += ['',] * max(len(gas_payments)-len(pmt_type), len(maintenance_payments)-len(pmt_type))
        print(72, len(pmt_dates))
        calculated_pmt_percentages += [0,] * max(len(gas_payments)-len(pmt_type), len(maintenance_payments)-len(pmt_type))
        print(73, len(calculated_pmt_percentages))
        pmt_amounts += [0,] * max(len(gas_payments)-len(pmt_type), len(maintenance_payments)-len(pmt_type))
        print(74, len(pmt_amounts))
        cumulative_pmt_percent += [0,] * max(len(gas_payments)-len(pmt_type), len(maintenance_payments)-len(pmt_type))
        print(75, len(cumulative_pmt_percent))
        if len(gas_payments) < len(pmt_type):
            print(1)
            gas_payments += [0,] * max(len(gas_payments)-len(pmt_type), len(maintenance_payments)-len(pmt_type))
            print(76, len(gas_payments))
        if len(maintenance_payments) < len(pmt_type):
            print(2)
            maintenance_payments += [0,] * max(len(gas_payments)-len(pmt_type), len(maintenance_payments)-len(pmt_type))
            print(77, len(maintenance_payments))
        print(78)
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
        "Cumulative Percentage":cumulative_pmt_percent,
        "Maintenance Fees":maintenance_payments,
        "Gas Fees":gas_payments
    }
    return payemnts_schedule