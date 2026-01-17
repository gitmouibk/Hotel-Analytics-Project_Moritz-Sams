# Hotel Analytics Project

## Project Goal
The goal of this project is to analyze hotel review data from TripAdvisor for Las Vegas hotels and to extract meaningful insights about hotel performance, customer behavior, and review patterns.
The project combines descriptive analytics, visual data storytelling, relationship exploration, and score prediction in a structured and reproducible way.

*A menu-driven Python program (main.py) allows the user to run the analyses interactively.*

## Folder Structure
```text
Hotel_Analytics_Project/
├── main.py
├── README.md
├── cleaning.py
├── performance.py
├── visualization.py
├── relationship.py
├── prediction.py
├── Part1_Basic_Exploration.ipynb
├── Part2_Business_Questions.ipynb
├── Part3_Forecasting.ipynb
├── plots/
└── report/
```

## How to Run the Program

### 1. Install Required Libraries

Make sure Python is installed, then run:
pip install pandas numpy matplotlib scikit-learn

### 2. Run the Main Program

Open a terminal in the project folder and run:
python main.py

### 3. Use the Menu

After starting the program, a menu with five options appears:
1. Program Info – explains the project and dataset
2. Analytics – hotel performance, visuals, and relationships
3. Hotel Score Prediction – linear regression model results
4. Help – instructions on how to use the program
5. Exit – closes the program

Enter a number (1–5) and press Enter to navigate.

## What Outputs Are Produced

### Console Outputs:
- Reviewer overview
- Traveler type overview 
- Top hotels by average Score
- Top hotels for Europe-only reviews
- Bottom hotels with major amenities
- Casino vs. non-casino Score comparison
- Model performance table
- Regression coefficients
- Section summaries for each menu section

### Visual Outputs

Plots are automatically saved in the plots/ folder:
- Histogram of review Scores
- Boxplot of Score by traveler type
- Boxplot of number of rooms by hotel stars
- Scatter plot of actual vs. predicted Score
- Correlation heatmap of numeric variables
- Relationship between years of being a member and helpful votes

### Model Outputs

- Train/test split results
- Linear regression model
- R² score (model explanatory power)
- Mean Squared Error (prediction error)
- Coefficient table (feature importance)

## How This Project Is Structured
- Notebooks are used for exploration
- .py files contain reusable functions
- main.py connects everything in a menu-driven program
- plots/ stores all generated figures
- report/ contains the final written analysis