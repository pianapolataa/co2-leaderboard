# CO2 Leaderboard

## Overview
CO2 Leaderboard is a full-stack application that tracks and visualizes carbon emissions across different states. It provides insights into historical trends, future predictions, and regional comparisons to monitor progress toward environmental goals.

## Tech Stack
- **Backend:** Flask, Pandas
- **Frontend:** React, TypeScript, Material-UI
- **Machine Learning:** PyTorch for neural network-based predictions

## Installation & Setup
### **Backend Setup**
1. Install dependencies:
   ```bash
   pip install flask flask-cors pandas torch torchvision torchmetrics
   ```
2. Run the Flask API:
   ```bash
   python app.py
   ```

### **Frontend Setup**
1. Navigate to the frontend directory (if applicable) and install dependencies:
   ```bash
   npm install
   ```
2. Start the frontend:
   ```bash
   npm run dev
   ```

## File Descriptions
### **Backend Files:**
- `app.py` - Flask API to serve emissions data.
- `co2pred.py` - PyTorch model for CO2 emission predictions.
- `test_viz.py` - Visualization script to test and compare model results.

### **Frontend Files:**
- `App.tsx` - Main React application file.
- `main.tsx` - Renders the React app.
- `Dashboard.tsx` - Displays US map of emissions.
- `USMap.tsx` - Interactive map for emissions data.
- `ChartPage.tsx` - Displays emissions trends for selected states.
- `EmissionChart.tsx` - Chart component for emissions visualization.

### **Data & Predictions:**
- `5 Year Predictions.csv` - Predicted CO2 emissions for future years.
- `table1_train.xlsx` - Training dataset for ML model.
- `table1_test.xlsx` - Test dataset for ML model.
- `table1_copy.xlsx` - Additional dataset for predictions.

## How to Use
1. **View emissions data** by running the Flask API (`app.py`).
2. **Interact with the dashboard** by starting the React frontend.
3. **Analyze predictions** using `co2pred.py` to generate forecasts.
4. **Explore visualizations** via the `ChartPage.tsx` and `EmissionChart.tsx` components.

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

