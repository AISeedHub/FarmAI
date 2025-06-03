# FarmAI

## MongoDB SSH Connection for Data Analysis

This project demonstrates how to connect to a MongoDB database through an SSH tunnel and load data into pandas DataFrames for analysis.

### Prerequisites

- Python 3.6+
- Jupyter Notebook
- Access to a MongoDB server via SSH
- Required Python packages (install with `pip install -r requirements.txt`):
  - pandas
  - pymongo
  - sshtunnel
  - python-dotenv
  - matplotlib (optional, for visualization)
  - seaborn (optional, for visualization)

### Setup

1. Clone this repository
2. Copy the `.env.example` file to `.env` and update it with your actual connection details:
   ```
   cp .env.example .env
   ```
3. Edit the `.env` file with your MongoDB and SSH connection parameters

### Usage

1. Open the Jupyter notebook:
   ```
   jupyter notebook notebooks/data_processing.ipynb
   ```
2. Run the cells in the notebook to:
   - Connect to MongoDB via SSH
   - Load data into a pandas DataFrame
   - Analyze and visualize the data

### Features

- Secure connection to MongoDB through SSH tunnel
- Environment variable management for sensitive credentials
- Flexible authentication methods (password or key-based)
- Error handling for connection issues
- Example data analysis and visualization code

### Troubleshooting

- If you encounter connection issues, verify your SSH and MongoDB credentials in the `.env` file
- Ensure your SSH server allows tunneling to the MongoDB port
- Check that the MongoDB server is running and accessible from the SSH server
