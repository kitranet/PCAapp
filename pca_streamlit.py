import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sidebar to select PCA
st.sidebar.title("PCA Settings")
pca_enabled = st.sidebar.checkbox("Enable PCA")

st.title(" Analyze & Amplify With PCA : Uncover Insights to Maximize Impact ")
# File upload functionality
uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV File", type=['xlsx', 'csv'])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Please upload a valid Excel or CSV file.")
            st.stop()

        # Filter numerical columns for PCA
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # If PCA is enabled
        if pca_enabled:
            # Select columns for PCA
            pca_columns = st.sidebar.multiselect("Select columns for PCA", numerical_columns)
            
            if pca_columns:
                # Filter the DataFrame
                pca_df = df[pca_columns]
                
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(pca_df)
                
                # Perform PCA
                pca = PCA()
                pca_result = pca.fit_transform(scaled_data)
                
                # Create PCA DataFrame
                pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(pca_result.shape[1])])
                
                # Concatenate PCA DataFrame with selected columns
                pca_df = pd.concat([df[pca_columns], pca_df], axis=1)
                
                # Show PCA DataFrame
                st.subheader("PCA Results")
                st.dataframe(pca_df)
                
                # Plot explained variance ratio
                plt.figure(figsize=(8, 6))
                plt.bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_)
                plt.xlabel("Principal Component")
                plt.ylabel("Explained Variance Ratio")
                plt.title("Explained Variance Ratio by Principal Component")
                st.pyplot(plt)
                
                # Show variance explained
                st.write("Variance explained by each component:")
                st.write(pca.explained_variance_ratio_)
            else:
                st.warning("Please select columns for PCA.")
        else:
            st.write("PCA is disabled.")

        # Show original DataFrame
        st.subheader("Original DataFrame")
        st.dataframe(df)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload an Excel or CSV file to proceed.")
