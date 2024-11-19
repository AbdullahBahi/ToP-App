from flask import Flask, request, jsonify

from utils import calculate_installments, get_units_data

app = Flask(__name__)

@app.route('/calculate_installments', methods=['POST'])
def calculate_installments_api():
    try:
        # Extract data from request
        data = request.json

        # Required fields
        unit_code = data['unit_code']
        tenor_years = data['tenor_years']
        payment_frequency = data['payment_frequency']
        contract_date = data['contract_date']
        input_pmts = data['input_pmts']
        contract_date = data['contract_date']
        input_pmts = {int(k):v for k, v in input_pmts.items()}

        # Get unit info
        unit_info = get_units_data(unit_code)

        # Call the calculate_installments function based on the selected project
        payment_schedule = calculate_installments(
            unit_info=unit_info,
            tenor_years=tenor_years,
            payment_frequency=payment_frequency,
            contract_date=contract_date,
            input_pmts=input_pmts,
        )
        return jsonify(payment_schedule), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
