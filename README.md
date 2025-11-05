Our final project idea is to implement a decision tree algorithm to ultimately use as a predictive model for determining the winner of the 2026 League of Legends Worlds Championship. We chose this project because it allows us to apply concepts we already know how to implement, such as decision trees and data analysis and code them in Rust while also having a real world application.

We will utilize the CSV dataset to get the past LoL match statistics, such as historical win rates, objective control, and early-game gold leads. This data will be processed and stored in memory-efficient structures, potentially leveraging the ndarray crate for fast numerical operations, setting the stage for classification.For the decision tree, we will define recursive nodes and tree structures. The train() function will recursively seek the best feature split by calculating Gini Impurity or information gain, effectively acting as the project's development roadmap. The final predict() function will be used to classify match-up data to predict the outcome of games leading up to the 2026 final.

Checkpoint Goals:
Checkpoint 1: Complete data ingestion setup using the csv database to successfully read and structure the data. Define all core tree data structures and start implementing the Gini impurity calculation function.

Checkpoint 2: Implement the full recursive train() function to build the tree structure. Complete and test the predict() function to classify new data points, achieving a functionally complete core algorithm."


The primary challenge will be ensuring memory safety and performance when handling the recursive ownership and borrowing required for building the tree structure in Rust. We also think that optimizing the split search for continuous features within large statistical datasets will be complicated. Finally, integrating and cleaning the diverse LoL data will require robust error handling.
