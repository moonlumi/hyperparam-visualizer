import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

def report_to_dataframe(report_dict):
    return pd.DataFrame(report_dict).transpose()

def plot_classification_report(report_df):
    fig, ax = plt.subplots(figsize=(10, 5))  # Set figure size
    ax.axis('off')  # Hide axes

    # Create a table and add it to the figure
    table = ax.table(cellText=report_df.values,
                     colLabels=report_df.columns,
                     rowLabels=report_df.index,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Scale table

    plt.title('Classification Report')
    return fig

def plot_precision_recall_curve(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    return plt

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    return plt


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    fig.colorbar(cax)
    
    # Set axis labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='black')

    return fig

def plot_class_distribution(y):
    plt.figure()
    class_counts = y.value_counts()
    plt.bar(class_counts.index, class_counts.values, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    return plt

def plot_feature_importance(coefficients, feature_names):
    plt.figure()
    plt.barh(feature_names, coefficients, color='salmon')
    plt.xlabel('Coefficient Value')
    plt.title('Feature Importance')
    return plt

def main():
    st.title("Classification Models on Alzheimer Dataset")

    

    
    # Sidebar for hyperparameter input
    st.sidebar.title("Model Hyperparameters")
    st.sidebar.subheader("Logistic Regression")

    # Hyperparameters for Logistic Regression


    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

    model = LogisticRegression()
    hyperp = model.get_params()

    key_dict={}
    for key,values in hyperp.items():
        check=False
        if isinstance(values,str):

            if values.isdigit():
                # values = int(values)
                check=True
        save_key=key
        # st.text(key)
        values = st.sidebar.text_input(f"Enter the {key} value",f"{values}")
        key_dict[key]=values
        if values=="None":
            key_dict[key]=None
        if values=="False":
            key_dict[key]=False
        if values=="True":
            key_dict[key]=True
        
        # st.text(key)
        
        try:
            float(values)
            # st.text("Float")
            key_dict[key]= float(values)
        except:
            pass

        # st.text(key_dict)

    
    # Load the dataset and split it
    uploaded_file = st.file_uploader("Choose a CSV file",type="csv")
    if uploaded_file is not None:
        alzheimer_data = pd.read_csv(uploaded_file)

    X= alzheimer_data.drop(["Diagnosis","DoctorInCharge"],axis=1)
    y = alzheimer_data["Diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize the Linear Regression model
    model = LogisticRegression(C=key_dict["C"],
                               class_weight=key_dict["class_weight"],
                               dual=key_dict["dual"],
                               fit_intercept=key_dict["fit_intercept"],
                               penalty=key_dict["penalty"],
                               intercept_scaling=key_dict["intercept_scaling"],
                               max_iter=int(key_dict["max_iter"]),
                               multi_class=key_dict["multi_class"],
                               n_jobs=key_dict["n_jobs"],
                               random_state=key_dict["random_state"],
                               solver=key_dict["solver"],
                               tol=key_dict["tol"],
                               verbose=int(key_dict["verbose"]),
                               warm_start=key_dict["warm_start"])

    # Train the model
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    
    col1, col2 = st.columns(2)

    with col1:
        # Display the results
        st.write("### Accuracy:", accuracy_score(y_test,y_preds))
        st.write('### Classification Report: ')
        class_report=classification_report(y_test,y_preds)
        # st.code(class_report,language="text")

        # Convert classification report to DataFrame
        report_df = report_to_dataframe(classification_report(y_test, y_preds, output_dict=True))

        # Plot and display the classification report
        fig = plot_classification_report(report_df)
        st.pyplot(fig)
        
        # Plotting Precision-Recall curve
        st.write("### Precision-Recall curve")
        y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
        pr_fig = plot_precision_recall_curve(y_test, y_probs)
        st.pyplot(pr_fig)

        # Plotting Class Distribution
        st.write("### Class Distribution")
        dist_fig = plot_class_distribution(y)
        st.pyplot(dist_fig)
    
    with col2:
        st.write("### Confusion Matrix")
        con_mat = confusion_matrix(y_test,y_preds)
        # st.code(con_mat,language="text")

        class_names = list(map(str, np.unique(y)))

        # Plotting Confusion Matrix
        fig = plot_confusion_matrix(con_mat, class_names)
        st.pyplot(fig)
    
        # Plotting ROC curve
        st.write("### ROC Curve")
        y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
        roc_fig = plot_roc_curve(y_test, y_probs)
        st.pyplot(roc_fig)
    
        # Plotting Feature Importance
        st.write("### Feature Importance")
        coefficients = model.coef_[0]  # Coefficients from the trained model
        feature_names = X.columns
        feat_fig = plot_feature_importance(coefficients, feature_names)
        st.pyplot(feat_fig)


if __name__ == "__main__":
    main()
