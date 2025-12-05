use std::error::Error;
// Removed unused HashMap import

// --- 1. CORE DATA STRUCTURES ---

/// Represents a single decision tree node.
/// We use 'Box<Node>' for recursive structures to manage memory easily.
#[derive(Debug)]
pub struct Node {
    /// The index of the feature (column) we split on.
    pub feature_index: Option<usize>,
    /// The value we use as a threshold for the split (e.g., a number or a category string).
    pub threshold: Option<String>, 
    /// The final class prediction if this node is a leaf (e.g., "Warrior").
    pub prediction: Option<String>,
    /// The two branches of the tree: (left_child, right_child).
    pub children: (Option<Box<Node>>, Option<Box<Node>>),
}

/// The main structure to hold all our raw data.
#[derive(Debug)]
pub struct Dataset {
    /// A vector of feature vectors. Each feature value is stored as a String.
    pub features: Vec<Vec<String>>,
    /// A vector of the class labels (targets). This is the 'Class' column in your LoL data.
    pub targets: Vec<String>,
    pub num_samples: usize,
    pub num_features: usize,
}

// --- 2. DATA LOADING LOGIC (Checkpoint 1 Goal) ---

impl Dataset {
    /// Loads the data from a simple CSV file. We assume the last column is the target.
    pub fn load_csv(filepath: &str) -> Result<Self, Box<dyn Error>> {
        println!("Loading LoL Champions dataset from: {}", filepath);
        
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true) // We expect the first row to be headers
            .from_path(filepath)?;
        
        let mut all_features: Vec<Vec<String>> = Vec::new();
        let mut all_targets: Vec<String> = Vec::new();
        let mut feature_count = 0;

        // Iterate through each record (row) in the CSV file
        for record_result in reader.records() {
            let record = record_result?; // Unwrap the successful record
            let mut current_row_features: Vec<String> = Vec::new();

            // Calculate how many total columns we have in this row
            let total_columns = record.len();
            
            // --- Separate Features from Target ---
            // Iterate over all fields in the row
            for (index, field) in record.iter().enumerate() {
                if index == total_columns - 1 {
                    // This is the LAST column ('Class'), so it's the target label
                    all_targets.push(field.to_string());
                } else {
                    // All other columns are features (Style, DamageType, etc.)
                    // We store them as simple strings for now to handle categories.
                    current_row_features.push(field.to_string());
                }
            }
            
            // Store the processed features for this row
            feature_count = current_row_features.len();
            all_features.push(current_row_features);
        }

        let sample_count = all_features.len();

        if sample_count == 0 {
            return Err(Box::from("Loaded dataset is empty. Check file content."));
        }

        println!("Successfully loaded {} champion samples with {} features each.", sample_count, feature_count);

        // Return the final Dataset structure
        Ok(Dataset {
            features: all_features,
            targets: all_targets,
            num_samples: sample_count,
            num_features: feature_count,
        })
    }
}

// --- 3. GINI IMPURITY CALCULATION (Will be added in next step) ---
// This section is currently empty, reserving it for Checkpoint 2.

// --- 4. MAIN EXECUTION ---

fn main() -> Result<(), Box<dyn Error>> {
    println!("--- LoL Decision Tree Checkpoint 1: Data Setup and Structures ---");

    // 1. Attempt to load the test data (lol_champions.csv)
    // The program will exit with an error message if the file is not found.
    let _dataset = match Dataset::load_csv("lol_champions.csv") {
        Ok(d) => d,
        Err(e) => {
            eprintln!("\nERROR: Data loading failed. Make sure 'lol_champions.csv' is in your project root.");
            return Err(e);
        }
    };

    // 2. Show the definition of the Node structure (the tree's building block)
    // This confirms the data containers are ready for the training logic.
    let root_node = Node {
        feature_index: None,
        threshold: None,
        prediction: None,
        children: (None, None), // No children yet
    };

    println!("\n--- Decision Tree Node Structure Defined ---");
    println!("The fundamental Node structure, ready for the splitting logic:\n{:?}", root_node);

    Ok(())
}