# Food Price Prediction System

A full-stack application for predicting food commodity prices using machine learning (XGBoost) with a React frontend and FastAPI backend.

## 📋 Table of Contents
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)

## 📁 Project Structure

```
project-root/
├── backend/
│   ├── main.py                    # FastAPI application entry point
│   ├── requirements.txt           # Python dependencies
│   ├── analytics/                 # Analytics and forecasting modules
│   ├── saved_models/              # Trained ML models
│   └── my_food_prices_avg.csv     # Historical price data
└── frontend/
    ├── package.json               # Node.js dependencies
    ├── src/                       # React source files
    ├── public/                    # Static assets
    └── node_modules/              # Installed npm packages (auto-generated)
```

## 🔧 Prerequisites

Before you begin, ensure you have the following installed on your system:

### Required Software
- **Python 3.8+** - [Download Python](https://www.python.org/downloads/)
- **Node.js 16+** and **npm** - [Download Node.js](https://nodejs.org/)
- **Git** (optional) - For cloning the repository

### Verify Installation
Open your terminal and run:

```bash
# Check Python version
python --version
# or
python3 --version

# Check Node.js version
node --version

# Check npm version
npm --version
```

## 📦 Installation

### Step 1: Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd <project-folder>

# Or download and extract the ZIP file
```

### Step 2: Set Up the Backend (Python)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. **(Recommended)** Create a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   This will install all required packages including:
   - FastAPI
   - Uvicorn
   - XGBoost
   - Pandas
   - NumPy
   - Scikit-learn
   - Joblib
   - And other dependencies

4. Verify installation:
   ```bash
   pip list
   ```

### Step 3: Set Up the Frontend (React)

1. Open a **new terminal window/tab** and navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install npm dependencies:
   ```bash
   npm install
   ```

   This will install all required packages including:
   - React
   - Vite
   - Axios
   - React Router
   - And other dependencies

3. Verify installation:
   ```bash
   npm list --depth=0
   ```

## 🚀 Running the Application

You need to run **both the backend and frontend** simultaneously in separate terminal windows.

### Terminal 1: Start the Backend Server

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Activate the virtual environment (if you created one):
   ```bash
   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

4. You should see output similar to:
   ```
   INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
   INFO:     Started reloader process [xxxxx] using WatchFiles
   INFO:     Started server process [xxxxx]
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   ```

5. The backend API is now running at: **http://localhost:8000**
   - API Documentation: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

### Terminal 2: Start the Frontend Development Server

1. Open a **new terminal window/tab**

2. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

3. Start the React development server:
   ```bash
   npm run dev
   ```

4. You should see output similar to:
   ```
   VITE v4.x.x  ready in xxx ms

   ➜  Local:   http://localhost:5173/
   ➜  Network: use --host to expose
   ➜  press h to show help
   ```

5. The frontend is now running at: **http://localhost:5173**

### Accessing the Application

1. Open your web browser
2. Navigate to: **http://localhost:5173**
3. The application should load and be able to communicate with the backend API


## 🛠️ Troubleshooting

### Backend Issues

**Problem: `uvicorn: command not found`**
```bash
# Solution: Make sure virtual environment is activated and uvicorn is installed
pip install uvicorn
```

**Problem: `ModuleNotFoundError: No module named 'fastapi'`**
```bash
# Solution: Install requirements again
pip install -r requirements.txt
```

**Problem: Port 8000 already in use**
```bash
# Solution: Use a different port
uvicorn main:app --reload --port 8001
```

**Problem: Models not loading**
```bash
# Solution: Ensure saved_models/ directory exists with .pkl files
# Re-run the training script if needed
```

### Frontend Issues

**Problem: `npm: command not found`**
```bash
# Solution: Install Node.js from https://nodejs.org/
```

**Problem: `Failed to resolve import`**
```bash
# Solution: Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Problem: Port 5173 already in use**
```bash
# Solution: Kill the process or use a different port
# On Windows: taskkill /F /IM node.exe
# On macOS/Linux: lsof -ti:5173 | xargs kill -9

# Or specify a different port
npm run dev -- --port 3000
```

**Problem: CORS errors when calling API**
```bash
# Solution: Check that backend is running on http://localhost:8000
# and that CORS is properly configured in main.py
```

### Common Issues

**Problem: Virtual environment not activating**
```bash
# Windows: Try using PowerShell with execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# macOS/Linux: Check permissions
chmod +x venv/bin/activate
```

**Problem: Different Python/Node versions**
```bash
# Check versions match requirements
python --version  # Should be 3.8+
node --version    # Should be 16+
```

## 📝 Development Notes

### Hot Reload
- **Backend**: The `--reload` flag enables auto-restart when Python files change
- **Frontend**: Vite automatically hot-reloads when you save changes to React files

### Environment Variables
If needed, create `.env` files:

**backend/.env**
```env
DATABASE_URL=your_database_url
API_KEY=your_api_key
```

**frontend/.env**
```env
VITE_API_URL=http://localhost:8000
```

### Stopping the Servers
- Press `CTRL + C` in each terminal window to stop the servers

## 🎯 Quick Start Summary

```bash
# Terminal 1 - Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Terminal 2 - Frontend  
cd frontend
npm install
npm run dev

# Open browser to http://localhost:5173
```

---

**Happy Coding! 🚀**