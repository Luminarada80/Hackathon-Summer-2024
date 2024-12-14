import os
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

atac_data_path = 'Hackathon2024.ATAC.txt'
rna_data_path = 'Hackathon2024.RNA.txt'

training_data_path = 'Hackathon2024.Training.Set.Peak2Gene.Pairs.txt'

testing_data_path = 'Hackathon2024.Testing.Set.Peak2Gene.Pairs.txt'

cis_reg_score_path = 'cell_type_specific_cis_regulatory_CD14 Mono.txt'

tss_difference_path = 'Training_PeakMidpoint_TSS_Difference.txt'
test_tss_difference_path= 'Test_PeakMidpoint_TSS_Difference.txt'

peak_accessibility_path = 'peak_accessibility.txt'
gc_content_path = 'GC_content.txt'
cv_peak_path = 'cv_peak.txt'


# ----- FUNCTION TO PARSE THE TRAINING/TESTING PREDICTION FILES -----
def parse_prediction_file(prediction_file_path):
    """
    Reads the information from one of the prediction data files (training or testing)
    and stores it in a dictionary.

    Dictionary format:\n
    - `data_dict[peak_gene_pair] = {'variable1':data, 'variable2':data, etc.}`

    :param prediction_file_path:
    :return: peak_list, gene_list, data_dict
    """
    data_dict = {}
    gene_list = []
    peak_list = []
    with open(prediction_file_path, 'r') as prediction_file:
        next(prediction_file)
        for line in prediction_file:
            line = line.strip().split('\t')
            peak = line[0]
            gene = line[1]
            pair = line[2]
            peak2gene = line[3]

            peak_list.append(peak)
            gene_list.append(gene)

            if pair not in data_dict.keys():
                data_dict[pair] = {
                    'peak2gene': peak2gene,
                    'cis_reg_score': 0,
                    'pearson_p_val': 0,
                    'seq_len': 0,
                    'tss_difference': 0,
                    'gc_content': 0,
                    'cv_peak': 0,
                    'peak_accessibility': 0
                    }
                
    return peak_list, gene_list, data_dict


# ----- FUNCTIONS TO ADD DATA TO THE DICTIONARIES -----

def add_from_region_to_dict(data_type, data_file_path, data_dict, peak_list, gene_list):
    """
    Adds data to the dictionary from files containing the peak followed by the data
    :param data_type:
    :param data_file_path:
    :param data_dict:
    :param peak_list:
    :param gene_list:
    :return:
    """
    with open(data_file_path, 'r') as data_file:
        next(data_file)
        for line in data_file:
            line = line.strip().split('\t')
            peak = line[0]
            if peak in peak_list:
                peak_index = peak_list.index(peak)
                gene = gene_list[peak_index]
                pair = peak + '_' + gene
                data = line[1]

                if data_type not in data_dict[pair]:
                    data_dict[pair][data_type] = 0

                data_dict[pair][data_type] = data


def add_cis_reg_score_to_dict(cis_reg_score_path, data_dict):
    """
    Adds the cis-regulatory score to the data dictionary
    :param cis_reg_score_path:
    :param data_dict:
    :return:
    """
    # Opens the file in read-only mode
    with open(cis_reg_score_path, 'r') as cis_reg_score_file:

        #next(cis_reg_score_file) <-- Add this if your data file has a HEADER, it will skip it

        # Iterate through each line in the file
        for line in cis_reg_score_file:
            line = line.strip().split('\t') # Removes new lines and splits the dataset into columns based on tabs
            pair = line[0] + '_' + line[1] # This specifies that the pair corresponds to column 1 plus column 2

            # If the pair is in the dictionary, append the cis-regulatory score (column 3)
            if pair in data_dict:
                # Numbers are read from files as strings by default, convert to a number such as int or fload
                data_dict[pair]['cis_reg_score'] = float(line[2])


def add_pearson_corr_to_dict(rna_data_path, atac_data_path, gene_list, peak_list, data_dict):
    """
    Calculates the Pearson correlation coefficient between the gene expression and
    peak read depth for the pair and adds it to the data dictionary
    :param rna_data_path:
    :param atac_data_path:
    :param gene_list:
    :param peak_list:
    :param data_dict:
    :return:
    """
    # Adds the gene expression to a dataframe
    rna_df = pd.read_csv(rna_data_path, sep='\t')
    gene_expr_df = rna_df[rna_df.iloc[:, 0].isin(gene_list)]

    # Adds the peak depth to a dataframe
    atac_df = pd.read_csv(atac_data_path, sep='\t')
    peak_depth_df = atac_df[atac_df.iloc[:, 0].isin(peak_list)]

    # Calculates the Pearson Correlation between the peak-gene pairs
    results = []
    for gene, peak in zip(gene_list, peak_list):
        if gene in gene_expr_df['gene'].values and peak in peak_depth_df['peak'].values:

            # Extract the corresponding rows
            gene_row = gene_expr_df.loc[gene_expr_df['gene'] == gene].iloc[:, 1:].squeeze()
            peak_row = peak_depth_df.loc[peak_depth_df['peak'] == peak].iloc[:, 1:].squeeze()

            # Calculate Pearson correlation
            correlation, p_value = stats.pearsonr(gene_row, peak_row)

            # Store the result
            results.append({'Gene': gene, 'Peak': peak, 'Pearson_correlation': correlation, 'P_value': p_value})
        else:
            print(f"Warning: Gene {gene} or Peak {peak} not found in the DataFrames.")

    correlation_df = pd.DataFrame(results)

    # Creates a 'Key' column in the DataFrame that matches the dictionary keys
    correlation_df['Key'] = correlation_df['Peak'] + '_' + correlation_df['Gene']

    # Loops through the dictionary and updates it with the p-values
    for pair in data_dict.keys():
        if pair in correlation_df['Key'].values:
            # Get the corresponding p-value
            p_value = correlation_df.loc[correlation_df['Key'] == pair, 'P_value'].values[0]
            # Update the dictionary with the p-value
            data_dict[pair]['pearson_p_val'] = p_value


def add_seq_len_to_dict(data_dict):
    """
    Adds the sequence length to the data dictionary
    :param data_dict:
    :return:
    """
    for pair in data_dict.keys():
        peak = pair.split('_')[0]

        start = peak.split('-')[1]
        end = peak.split('-')[2]

        # Calculate the peak length
        peak_seq_len = int(end) - int(start)

        # Add the peak_sequence_length to the dictionary
        data_dict[pair]['seq_len'] = int(peak_seq_len)


def add_tss_difference_to_dict(tss_difference_path, data_dict):
    """
    Adds the TSS difference to the data dictionary.
    :param tss_difference_path: Path to the file containing TSS differences.
    :param data_dict: Dictionary to update with TSS difference data.
    :return: None
    """
    # Read the TSS difference file
    tss_df = pd.read_csv(tss_difference_path, sep='\t')
    
    # Iterate through the dataframe and add TSS difference to the dictionary
    for index, row in tss_df.iterrows():
        pair = row['pair']
        difference = row['difference']
        
        if pair in data_dict:
            data_dict[pair]['tss_difference'] = difference
        else:
            print(f"Warning: {pair} not found")


# ----- RANDOM FOREST FUNCTIONS -----
def random_forest_classifier(training_data_dict):
    """
    Uses the training data dictionary to train a random forest model to predict whether the
    label should be TRUE or FALSE.

    If you add more variables to training_data_dict, add them to the `X` variable

    :param training_data_dict:
    :return: random_forest_model
    """

    # Converts the training data dictionary to a pandas dataframe
    df = pd.DataFrame.from_dict(training_data_dict, orient='index')

    # Converts the TRUE/FALSE to a 1 or 0
    df['peak2gene'] = df['peak2gene'].astype('category')
    df['peak2gene'] = df['peak2gene'].apply(lambda x: 1 if x == 'TRUE' else 0)

    # Define which columns are data and which column is the target
    X = df.drop(columns=['peak2gene'])
    y = df['peak2gene']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    rf.fit(X_train, y_train)

    # Calculate the importance of each feature to the model
    importances = rf.feature_importances_
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)

    print(feature_importance)

    # Store the columns so that they can be matched later
    X_train_columns = X_train.columns

    return rf, X_train_columns


def lightgbm_classifier(training_data_dict):
    """
    Uses the data dictionary to train a gradient boosted model.

    If you end up adding more variables to training_data_dict, add them to the `X` variable.

    :param training_data_dict:
    :return: best_lgb_model, X_train_columns
    """
    # Converts the training data dictionary to a pandas dataframe
    df = pd.DataFrame.from_dict(training_data_dict, orient='index')

    # Converts the TRUE/FALSE to a 1 or 0
    df['peak2gene'] = df['peak2gene'].astype('category')
    df['peak2gene'] = df['peak2gene'].apply(lambda x: 1 if x == 'TRUE' else 0)

    # Converts everything to numeric
    df['gc_content'] = pd.to_numeric(df['gc_content'], errors='coerce')
    df['cv_peak'] = pd.to_numeric(df['cv_peak'], errors='coerce')
    df['peak_accessibility'] = pd.to_numeric(df['peak_accessibility'], errors='coerce')

    # Gets rid of any NA values
    df.dropna(inplace=True)

    # Define features and target that the model is trying to predict
    X = df.drop(columns=['peak2gene'])  # Use all columns except 'peak2gene' as features
    y = df['peak2gene']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter grid for LightGBM
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Initialize GridSearchCV with LightGBM
    grid_search = GridSearchCV(
        estimator=lgb.LGBMClassifier(random_state=42, verbose=-1),
        param_grid=param_grid,
        cv=5,
        n_jobs=6,
        verbose=-1,
        scoring='accuracy'
    )

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best model and its parameters
    print("Best parameters found: ", grid_search.best_params_)
    best_lgb_model = grid_search.best_estimator_

    # Evaluate on the test set
    y_pred = best_lgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy of the best model: ", accuracy)

    X_train_columns_gbm = X_train.columns

    return best_lgb_model, X_train_columns_gbm


def random_forest_predictions(test_data_dict, random_forest, X_train_columns):
    """
    Generate predictions using the trained Random Forest model.

    :param test_data_dict: Dictionary containing test data where keys are feature names and values are lists of feature values.
    :param random_forest: Trained Random Forest model.
    :param X_train_columns: List of feature names used during training.
    :return: Prediction probabilities.
    """

    true_pred_vals = []
    false_pred_vals = []
    for peak_gene, variable_gene in test_data_dict.items():
        # if 'cis_reg_score' in test_data_dict and 'seq_len' in test_data_dict and 'pearson_p_val' in test_data_dict:
        df_test = pd.DataFrame({
            'cis_reg_score': [variable_gene['cis_reg_score']],
            'seq_len': [variable_gene['seq_len']],
            'pearson_p_val': [variable_gene['pearson_p_val']],
            'tss_difference': [variable_gene['tss_difference']],
            'gc_content': [variable_gene['gc_content']],
            'cv_peak': [variable_gene['cv_peak']],
            'peak_accessibility': [variable_gene['peak_accessibility']],
        })

        # Ensure that df_test has the same columns as X_train_columns
        for col in X_train_columns:
            if col not in df_test.columns:
                df_test[col] = 0  # Or use another default value, e.g., NaN

        # Reorder the test data to match the training data
        X_new = df_test[X_train_columns]

        # Generate prediction probabilities from the model
        prediction_prob = random_forest.predict_proba(X_new)

        predicted_label = 'TRUE' if prediction_prob[0][1] > 0.5 else 'FALSE' # Sets the label based on the probability
        variable_gene['peak2gene'] = predicted_label # Sets the dictionary value to TRUE or FALSE for peak2gene

        # Appends the true/false prediction probability to a list for summary stats
        if prediction_prob[0][1] > 0.5:
            true_pred_vals.append(prediction_prob[0][1])
        else:
            false_pred_vals.append(prediction_prob[0][0])


    print(f'\t\tAverage TRUE prediction confidence: {statistics.mean(true_pred_vals)}')
    print(f'\t\t\tstdev: {statistics.stdev(true_pred_vals)}')
    print(f'\t\t\tmax: {max(true_pred_vals)}')
    print(f'\t\t\tmin: {min(true_pred_vals)}')

    print(f'\t\tAverage FALSE prediction confidence: {statistics.mean(false_pred_vals)}')
    print(f'\t\t\tstdev: {statistics.stdev(false_pred_vals)}')
    print(f'\t\t\tmax: {max(false_pred_vals)}')
    print(f'\t\t\tmin: {min(false_pred_vals)}')


def gradient_boosting_predictions(test_data_dict, model, X_train_columns_gbm):
    """
    Generate predictions using the trained Gradient Boosting model (XGBoost or LightGBM).

    :param test_data_dict: Dictionary containing test data where keys are feature names and values are lists of feature values.
    :param model: Trained Gradient Boosting model (XGBoost or LightGBM).
    :param X_train_columns: List of feature names used during training.
    :return: Prediction probabilities.
    """

    true_pred_vals_gbm = []
    false_pred_vals_gbm = []
    for peak_gene, single_test_data in test_data_dict.items():
        # Convert the test data dictionary to a pandas DataFrame
        df_test = pd.DataFrame({
            'cis_reg_score': [single_test_data['cis_reg_score']],
            'seq_len': [single_test_data['seq_len']],
            'pearson_p_val': [single_test_data['pearson_p_val']],
            'tss_difference': [single_test_data['tss_difference']],
            'gc_content': [single_test_data['gc_content']],
            'cv_peak': [single_test_data['cv_peak']],
            'peak_accessibility': [single_test_data['peak_accessibility']],
        })

        # Ensure that all features are numeric
        df_test['gc_content'] = pd.to_numeric(df_test['gc_content'], errors='coerce')
        df_test['cv_peak'] = pd.to_numeric(df_test['cv_peak'], errors='coerce')
        df_test['peak_accessibility'] = pd.to_numeric(df_test['peak_accessibility'], errors='coerce')

        # Handle any NA values
        df_test.fillna(0, inplace=True)

        # Ensure that df_test has the same columns as X_train_columns
        for col in X_train_columns_gbm:
            if col not in df_test.columns:
                df_test[col] = 0

        # Reorder the test data to match the training data
        X_new = df_test[X_train_columns_gbm]

        # Generate prediction probabilities from the model
        prediction_prob = model.predict_proba(X_new)

        predicted_label = 'TRUE' if prediction_prob[0][1] > 0.5 else 'FALSE'
        single_test_data['peak2gene'] = predicted_label

        # Appends the true/false prediction probability to a list for summary stats
        if prediction_prob[0][1] > 0.5:
            true_pred_vals_gbm.append(prediction_prob[0][1])
        else:
            false_pred_vals_gbm.append(prediction_prob[0][0])

    print(f'\t\tAverage TRUE prediction confidence: {statistics.mean(true_pred_vals_gbm)}')
    print(f'\t\t\tstdev: {statistics.stdev(true_pred_vals_gbm)}')
    print(f'\t\t\tmax: {max(true_pred_vals_gbm)}')
    print(f'\t\t\tmin: {min(true_pred_vals_gbm)}')

    print(f'\t\tAverage FALSE prediction confidence: {statistics.mean(false_pred_vals_gbm)}')
    print(f'\t\t\tstdev: {statistics.stdev(false_pred_vals_gbm)}')
    print(f'\t\t\tmax: {max(false_pred_vals_gbm)}')
    print(f'\t\t\tmin: {min(false_pred_vals_gbm)}')


def compare_model_predictions(test_data_dict, rf_model, gbm_model, X_train_columns_rf, X_train_columns_gbm):
    """
    Compare predictions from Random Forest and GBM and select the one with the highest confidence.

    :param test_data_dict: Dictionary containing test data where keys are feature names and values are lists of feature values.
    :param rf_model: Trained Random Forest model.
    :param gbm_model: Trained GBM model.
    :param X_train_columns_rf: List of feature names used during Random Forest training.
    :param X_train_columns_gbm: List of feature names used during GBM training.
    :return: List of final predictions with the highest confidence.
    """

    final_predictions = []

    for peak_gene, single_test_data in test_data_dict.items():
        # Prepare the data for both models
        df_test_rf = pd.DataFrame({col: [single_test_data.get(col, 0)] for col in X_train_columns_rf})
        df_test_gbm = pd.DataFrame({col: [single_test_data.get(col, 0)] for col in X_train_columns_gbm})

        # Make columns numeric
        df_test_gbm['gc_content'] = pd.to_numeric(df_test_gbm['gc_content'], errors='coerce')
        df_test_gbm['cv_peak'] = pd.to_numeric(df_test_gbm['cv_peak'], errors='coerce')
        df_test_gbm['peak_accessibility'] = pd.to_numeric(df_test_gbm['peak_accessibility'], errors='coerce')

        # Handle NA values
        df_test_gbm.fillna(0, inplace=True)

        # Predict probabilities for TRUE
        prob_rf = rf_model.predict_proba(df_test_rf)[:, 1][0]
        prob_gbm = gbm_model.predict_proba(df_test_gbm)[:, 1][0]

        # Compare the models and use the one with the higher confidence
        if prob_rf > prob_gbm:
            final_predictions.append(prob_rf if prob_rf > 0.5 else 1 - prob_rf)
            if prob_rf > 0.5:
                test_data_dict[peak_gene]['peak2gene'] = 'TRUE'
            else:
                test_data_dict[peak_gene]['peak2gene'] = 'FALSE'

        else:
            final_predictions.append(prob_gbm if prob_gbm > 0.5 else 1 - prob_gbm)
            if prob_gbm > 0.5:
                test_data_dict[peak_gene]['peak2gene'] = 'TRUE'
            else:
                test_data_dict[peak_gene]['peak2gene'] = 'FALSE'

    return final_predictions


def plot_all_data(training_data_dict):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(training_data_dict, orient='index')
    df['peak2gene'] = df['peak2gene'].astype('category')

    # Ensure all relevant columns are numeric
    df['cv_peak'] = pd.to_numeric(df['cv_peak'], errors='coerce')
    df['peak_accessibility'] = pd.to_numeric(df['peak_accessibility'], errors='coerce')
    df['gc_content'] = pd.to_numeric(df['gc_content'], errors='coerce')

    # Pair plot for a quick overview of the relationships
    sns.pairplot(df, hue='peak2gene', markers=["o", "s"])
    plt.suptitle('Pair Plot of cis_reg_score, seq_len, p_value, and tss_difference by Peak2Gene', y=1.02)
    plt.show()

    plt.figure(figsize=(20, 12))  # Increased figure size for better spacing

    # Box plot for cis_reg_score
    plt.subplot(2, 4, 1)
    sns.boxplot(x='peak2gene', y='cis_reg_score', data=df)
    plt.title('cis_reg_score by Peak2Gene')

    # Box plot for seq_len
    plt.subplot(2, 4, 2)
    sns.boxplot(x='peak2gene', y='seq_len', data=df)
    plt.title('seq_len by Peak2Gene')

    # Box plot for pearson_p_val (log scale due to potentially wide range)
    plt.subplot(2, 4, 3)
    sns.boxplot(x='peak2gene', y='pearson_p_val', data=df)
    plt.yscale('log')
    plt.title('pearson_p_val by Peak2Gene')

    # Box plot for tss_difference
    plt.subplot(2, 4, 4)
    sns.boxplot(x='peak2gene', y='tss_difference', data=df)
    plt.title('tss_difference by Peak2Gene')

    # Box plot for cv_peak
    plt.subplot(2, 4, 5)
    sns.boxplot(x='peak2gene', y='cv_peak', data=df)
    plt.title('cv_peak by Peak2Gene')

    # Box plot for peak_accessibility
    plt.subplot(2, 4, 6)
    sns.boxplot(x='peak2gene', y='peak_accessibility', data=df)
    plt.title('peak_accessibility by Peak2Gene')

    # Box plot for gc_content
    plt.subplot(2, 4, 7)
    sns.boxplot(x='peak2gene', y='gc_content', data=df)
    plt.title('gc_content by Peak2Gene')

    # Adjust the layout to provide more space between plots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # hspace for vertical spacing, wspace for horizontal spacing

    plt.show()


# ----- TRAINING PIPELINE -----
def train_on_test_data(training_data_path):
    print('----- TRAINING DATA -----')


    print(f'\tCreating training data dictionary...')
    # Dictionary structure: dict[peak] = {'peak2gene': 'TRUE', 'cis_reg_score': 0, 'seq_len': 0, 'pearson_p_val': 0}
    peak_list, gene_list, training_data_dict = parse_prediction_file(training_data_path)

    print(f'\tAdding GC content to dictionary...')
    add_from_region_to_dict('gc_content', gc_content_path, training_data_dict, peak_list, gene_list)

    print(f'\tAdding CV peak data to dictionary...')
    add_from_region_to_dict('cv_peak', cv_peak_path, training_data_dict, peak_list, gene_list)

    print(f'\tAdding peak accessibility to dictionary...')
    add_from_region_to_dict('peak_accessibility', peak_accessibility_path, training_data_dict, peak_list, gene_list)

    # ----- Adding data to the training peak-gene dictionary -----
    print('\tAdding peak sequence length to dictionary...')
    add_seq_len_to_dict(training_data_dict)

    print('\tAdding cis-regulatory potential score to dictionary...')
    add_cis_reg_score_to_dict(cis_reg_score_path, training_data_dict)

    print(f'\tAdding Pearson correlation to dictionary...')
    add_pearson_corr_to_dict(rna_data_path, atac_data_path, gene_list, peak_list, training_data_dict)

    print('\tAdding TSS difference to the dictionary...')
    add_tss_difference_to_dict(tss_difference_path, training_data_dict)


    # ----- Training the random forest classifier -----
    print('\tTraining the Random Forest Classifier...')
    random_forest, X_train_columns = random_forest_classifier(training_data_dict)

    print('\tTraining the gradient boosting machine model...')
    light_gbm_model, X_train_columns_gbm = lightgbm_classifier(training_data_dict)

    plot_all_data(training_data_dict)

    return random_forest, X_train_columns, light_gbm_model, X_train_columns_gbm


#----- PREDICTION PIPELINE -----
def make_predictions(test_data_path, random_forest, X_train_columns, light_gbm_model, X_train_columns_gbm):
    print('\n----- TEST DATASET -----')
    print('\tCreating the test data dictionary...')
    peak_list, gene_list, test_data_dict = parse_prediction_file(test_data_path)

    # ----- Adding data to the test peak-gene dictionary -----
    print(f'\tAdding GC content to dictionary...')
    add_from_region_to_dict('gc_content', gc_content_path, test_data_dict, peak_list, gene_list)

    print(f'\tAdding CV peak data to dictionary...')
    add_from_region_to_dict('cv_peak', cv_peak_path, test_data_dict, peak_list, gene_list)

    print(f'\tAdding peak accessibility to dictionary...')
    add_from_region_to_dict('peak_accessibility', peak_accessibility_path, test_data_dict, peak_list, gene_list)

    print('\tAdding peak sequence length to dictionary...')
    add_seq_len_to_dict(test_data_dict)

    print('\tAdding cis-regulatory potential score to dictionary...')
    add_cis_reg_score_to_dict(cis_reg_score_path, test_data_dict)

    print(f'\tAdding Pearson correlation to dictionary...')
    add_pearson_corr_to_dict(rna_data_path, atac_data_path, gene_list, peak_list, test_data_dict)

    print('\tAdding sequence length results to the dictionary...')
    add_seq_len_to_dict(test_data_dict)

    print('\tAdding TSS difference to the dictionary...')
    add_tss_difference_to_dict(test_tss_difference_path,test_data_dict)


    # ----- Random forest predictions -----
    print('\tMaking random forest predictions...')
    random_forest_predictions(test_data_dict, random_forest, X_train_columns)

    print(f'\tMaking random gradient predictions...')
    gradient_boosting_predictions(test_data_dict, light_gbm_model, X_train_columns_gbm)

    # Compare predictions and get the final list with highest confidence
    final_predictions = compare_model_predictions(test_data_dict, random_forest, light_gbm_model, X_train_columns,
                                                  X_train_columns_gbm)

    print(f'\n\tFinal prediction confidence:')
    print(f'\t\tAverage TRUE prediction confidence: {statistics.mean(final_predictions)}')
    print(f'\t\t\tstdev: {statistics.stdev(final_predictions)}')
    print(f'\t\t\tmax: {max(final_predictions)}')
    print(f'\t\t\tmin: {min(final_predictions)}')

    # ----- Writing predictions to the predictions file -----
    print(f'\tWriting predictions to the predictions file...')
    os.makedirs('../prediction', exist_ok=True)
    with open('../prediction/prediction.csv', 'w') as prediction_file:
        prediction_file.write('peak,gene,Pair,Peak2Gene\n')
        for pair in test_data_dict.keys():
            peak = pair.split('_')[0]
            gene = pair.split('_')[1]
            prediction_file.write(f'{peak},{gene},{pair},{test_data_dict[pair]["peak2gene"]}\n')


if __name__ == "__main__":
    random_forest, X_train_columns, light_gbm_model, X_train_columns_gbm = train_on_test_data(training_data_path)
    make_predictions(testing_data_path, random_forest, X_train_columns, light_gbm_model, X_train_columns_gbm)