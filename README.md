🚕 Optimizing Ride-Sharing Matching Using Graph Theory
📌 Overview

This project models a ride-sharing system (similar to Uber) using Graph Theory and optimization techniques.

Drivers and passengers are represented as nodes in a graph, while ride offers form weighted edges. The system evaluates multiple strategies to efficiently match passengers with drivers based on price, wait time, distance, and driver rating.

🎯 Objectives
Minimize passenger wait time
Optimize ride pricing
Improve driver-passenger matching efficiency
Simulate real-world ride request scenarios
Apply graph theory to solve practical optimization problems
🧠 Key Concepts Used
Bipartite Graphs (Drivers ↔ Passengers)
Weighted Matching
Multi-Objective Optimization
Cost Functions
Ranking & Filtering Algorithms
🚀 Features
🔹 Core System
Graph-based ride matching
Multiple assignment strategies:
Baseline (nearest driver)
Marketplace (strict / relaxed / all offers)
Performance comparison using metrics
🔹 Live Ride Simulation
Enter pickup & drop locations (coordinates or names)
Choose preference:
Cheap
Fast
Premium
Balanced
Select vehicle type:
Economy / Standard / Premium
Enter budget range
🔹 Smart Filtering
Budget-based driver filtering
Driver acceptance rules
Vehicle compatibility
🔹 Intelligent Recommendations
Top 3 driver suggestions
Tags:
Best Overall
Fastest Pickup
Highest Rated
Explanation for each recommendation
🔹 Visualization
Strategy comparison charts:
Matched vs Unmatched passengers
Price comparison
Wait time comparison
Total cost comparison
📊 Example Output

The system provides:

🚕 Best driver match
📋 List of all drivers within budget
🏆 Top 3 recommended drivers
📈 Graph visualizations
🧮 Cost Function

The system uses a weighted scoring function:

Score = (0.5 × Price) + (0.3 × Wait Time) − (0.2 × Rating)

👉 This balances:

Lower price ✅
Lower wait time ✅
Higher driver rating ✅
🛠 Tech Stack
Python
Pandas
NumPy
NetworkX
Matplotlib
📂 Project Structure
GRAPH_THEORY_PROJECT/
│
├── data/                # Input datasets
├── outputs/             # Generated CSVs & charts
├── src/
│   ├── main.py          # Core matching logic
│   ├── live_request.py  # Interactive ride request system
│   ├── visualize.py     # Charts & graphs
│   └── demo.py          # Demo script
│
├── README.md
├── requirements.txt
└── .gitignore
▶️ How to Run
1. Install dependencies
pip install -r requirements.txt
2. Run main analysis
python3 src/main.py
3. Run live ride simulation
python3 src/live_request.py
4. Generate visualizations
python3 src/visualize.py
📈 Key Insights
Cheapest drivers often have higher wait times
Strict filtering reduces matches but improves efficiency
Relaxed marketplace improves balance between cost and availability
Multi-criteria ranking produces better real-world results
🔮 Future Improvements
Real-time driver location updates
Map-based visualization (Google Maps integration)
Dynamic surge pricing
Machine learning-based predictions
Mobile app integration
👨‍💻 Author

Pratham Santosh Mavle
Master’s in Computer Science – University of North Texas

💡 Key Takeaway

This project demonstrates how graph theory + optimization can be applied to solve real-world problems like ride-sharing, balancing cost, efficiency, and user preferences.
