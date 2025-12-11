use std::collections::HashMap;
use crate::dataset::Dataset; 

#[derive(Debug)]
pub struct Node {
    pub feature_index: Option<usize>,
    pub threshold: Option<String>,
    pub prediction: Option<String>,
    pub children: (Option<Box<Node>>, Option<Box<Node>>),
}

// Calculates how mixed the labels are in a set of rows.
fn gini_impurity(dataset: &Dataset, row_indices: &[usize]) -> f64 {
    if row_indices.is_empty() { return 0.0; }
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for &idx in row_indices {
        *counts.entry(dataset.targets[idx].as_str()).or_insert(0) += 1;
    }
    let n = row_indices.len() as f64;
    let mut impurity = 1.0; // Start at 1.0 and subtract p^2 for each class.
    for &count in counts.values() {
        let p = count as f64 / n;
        impurity -= p * p; // Gini is 1 - sum(p^2).
    }
    impurity
}

// Checks if all the labels in this chunk of data are the same.
fn all_same_class(dataset: &Dataset, row_indices: &[usize]) -> bool {
    if row_indices.is_empty() { return true; }
    let first = &dataset.targets[row_indices[0]];
    row_indices.iter().all(|&idx| &dataset.targets[idx] == first)
}

// Finds the most frequent label in this group of rows. Used for leaf predictions.
fn majority_class(dataset: &Dataset, row_indices: &[usize]) -> String {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for &idx in row_indices {
        *counts.entry(dataset.targets[idx].as_str()).or_insert(0) += 1;
    }
    let mut best_label = "";
    let mut best_count = 0usize;
    for (&label, &count) in counts.iter() {
        if count > best_count {
            best_count = count;
            best_label = label;
        }
    }
    best_label.to_string()
}

// Loops through all features and values to find the split that minimizes Gini.
fn find_best_split(
    dataset: &Dataset,
    row_indices: &[usize],
) -> Option<(usize, String, Vec<usize>, Vec<usize>)> {
    let mut best_feature = 0usize;
    let mut best_threshold = String::new();
    let mut best_impurity = f64::INFINITY; //need a high value
    let mut best_left = Vec::new();
    let mut best_right = Vec::new();

    if row_indices.is_empty() { return None; }
    // Try every single feature as the splitting column.
    for feat_idx in 0..dataset.num_features {
        let mut unique_vals: HashMap<&str, ()> = HashMap::new();
        for &row in row_indices {
            unique_vals.entry(dataset.features[row][feat_idx].as_str()).or_insert(());
        }

        for &val in unique_vals.keys() {
            let mut left_rows = Vec::new();
            let mut right_rows = Vec::new();
            // Split the rows and if left = feature == val, right = feature != val.
            for &row in row_indices {
                if dataset.features[row][feat_idx].as_str() == val {
                    left_rows.push(row);
                } else {
                    right_rows.push(row);
                }
            }

            if left_rows.is_empty() || right_rows.is_empty() { continue; }

            let g_left = gini_impurity(dataset, &left_rows);
            let g_right = gini_impurity(dataset, &right_rows);
            // Calculate the weighted average of the two Gini scores.
            let n = row_indices.len() as f64;
            let weighted = (left_rows.len() as f64 / n) * g_left + (right_rows.len() as f64 / n) * g_right;
            // Is this better than the best split so far?
            if weighted < best_impurity {
                best_impurity = weighted;
                best_feature = feat_idx;
                best_threshold = val.to_string();
                best_left = left_rows;
                best_right = right_rows;
            }
        }
    }

    if best_impurity == f64::INFINITY {
        None
    } else {
        Some((best_feature, best_threshold, best_left, best_right))
    }
}
    //builds the tree recursively.
pub fn build_tree(dataset: &Dataset, row_indices: &[usize], depth: usize, max_depth: usize) -> Node {
    if row_indices.is_empty() || all_same_class(dataset, row_indices) || depth >= max_depth {
        //Stop if pure, hit max depth, or out of data.
        return Node {
            feature_index: None, threshold: None,
            prediction: Some(if row_indices.is_empty() { "UNKNOWN".to_string() } else { majority_class(dataset, row_indices) }),
            children: (None, None),
        };
    }
    
    let split = find_best_split(dataset, row_indices);
    if split.is_none() {
        return Node {
            feature_index: None, threshold: None,
            prediction: Some(majority_class(dataset, row_indices)), 
            children: (None, None),
        };
    }

    let (feat_idx, threshold, left_rows, right_rows) = split.unwrap();
    // Recursively call for the next level down
    let left_child = build_tree(dataset, &left_rows, depth + 1, max_depth);
    let right_child = build_tree(dataset, &right_rows, depth + 1, max_depth);

    Node {  //node becomes an internal node
        feature_index: Some(feat_idx),
        threshold: Some(threshold),
        prediction: None,
        children: (Some(Box::new(left_child)), Some(Box::new(right_child))),
    }
}
impl Node {
    pub fn predict(&self, features: &[String]) -> String {
        if let Some(ref final_prediction) = self.prediction {
            return final_prediction.clone();
        }

        let feature_index_to_check = self.feature_index.expect("missing a feature index");
        let split_threshold = self.threshold.as_ref().expect("missing a split valu");

        // Get value from the input features
        let value_in_data = &features[feature_index_to_check]; 


        let (left_child_opt, right_child_opt) = &self.children;

        let next_child = if value_in_data == split_threshold {
            // If val matches threshold, go Left
            left_child_opt
        } else {
            right_child_opt
        };

        match next_child {
            Some(child_node) => {
                // If the child exists, recurse predict
                child_node.predict(features)
            },
            None => {
                "UNKNOWN".to_string()
            }
        }
    }
}