from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
app = Flask(__name__)


# Load the cleaned emissions data
data = pd.read_csv('/Users/jordansalagala/co2-leaderboard/cetbackend/backend/data/dataemissions.csv')
data = data.dropna(axis=1, how='all')
data = data.dropna(axis=0, how='all')  # Remove rows with all NaN values  # Remove columns with all NaN values

  # Enable CORS for all routes
CORS(app)



@app.route('/api/emissions', methods=['GET'])
def get_emissions():
    # Optional: Filter by state if needed
    state = request.args.get('state')
    
    if state:
        state_data = data[data['State'] == state].to_dict(orient='records')
        return jsonify(state_data)
    
    return jsonify(data.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)