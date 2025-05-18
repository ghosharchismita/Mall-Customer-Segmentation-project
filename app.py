import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🛍️ Mall Customers Segmentation App")
st.markdown("""
Welcome to the **Customer Segmentation App**, where you can:  
- Analyze your customer data interactively.  
- Apply clustering methods like **K-Means**, **Hierarchical Clustering**, and **DBSCAN**.  
- Gain valuable insights into customer groups with **profiler dashboards** and **dynamic visualizations**.  
            
This app has been built using the **Mall Customers Dataset** from **Kaggle**.
            
The app is user-friendly, with customizable options at every step!
""")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Mall_Customers.csv')
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    return df

df = load_data()

# Data Scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# Sidebar Navigation
st.sidebar.title("🔎 Navigation")
method = st.sidebar.radio("Choose a clustering method", ["K-Means", "Hierarchical", "DBSCAN"])
st.sidebar.write("Use the sliders below to adjust parameters as needed.")

# Data Preview Section
with st.expander("📊 View Dataset"):
    st.write("Here's a quick preview of the dataset being analyzed:")
    st.dataframe(df.head(10))

# Apply Clustering
if method == "K-Means":
    st.header("📍 K-Means Clustering")
    n_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5, help="Adjust the number of clusters.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster_KMeans'] = kmeans.fit_predict(df_scaled)

    # Metrics
    st.write("### Evaluation Metrics")
    sil_score = silhouette_score(df_scaled, df['Cluster_KMeans'])
    db_score = davies_bouldin_score(df_scaled, df['Cluster_KMeans'])
    st.metric(label="Silhouette Score", value=f"{sil_score:.3f}", help="Measures cluster separation (higher is better).")
    st.metric(label="Davies-Bouldin Index", value=f"{db_score:.3f}", help="Measures cluster compactness (lower is better).")

    # Visualization
    st.write("### Clustering Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Cluster_KMeans'], palette='viridis', ax=ax)
    ax.set_title(f"K-Means Clustering (K={n_clusters})")
    st.pyplot(fig)

elif method == "Hierarchical":
    st.header("📍 Hierarchical Clustering")
    linkage_method = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
    Z = linkage(df_scaled, method=linkage_method)
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5, help="Adjust the number of clusters.")
    df['Cluster_HC'] = fcluster(Z, t=n_clusters, criterion='maxclust')

    # Metrics
    st.write("### Evaluation Metrics")
    sil_score = silhouette_score(df_scaled, df['Cluster_HC'])
    db_score = davies_bouldin_score(df_scaled, df['Cluster_HC'])
    st.metric(label="Silhouette Score", value=f"{sil_score:.3f}")
    st.metric(label="Davies-Bouldin Index", value=f"{db_score:.3f}")

    # Dendrogram Visualization
    st.write("### Dendrogram")
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, ax=ax, color_threshold=0.5)
    ax.set_title(f"Hierarchical Clustering Dendrogram ({linkage_method.capitalize()} Method)")
    st.pyplot(fig)

elif method == "DBSCAN":
    st.header("📍 DBSCAN Clustering")
    eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=1.0, value=0.5, step=0.1, help="Controls the radius of the neighborhood.")
    min_samples = st.slider("Minimum Samples", min_value=1, max_value=10, value=5, help="Minimum number of points required to form a cluster.")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster_DBSCAN'] = dbscan.fit_predict(df_scaled)

    # Metrics
    st.write("### Evaluation Metrics")
    if len(set(df['Cluster_DBSCAN'])) > 1:
        sil_score = silhouette_score(df_scaled, df['Cluster_DBSCAN'])
        db_score = davies_bouldin_score(df_scaled, df['Cluster_DBSCAN'])
        st.metric(label="Silhouette Score", value=f"{sil_score:.3f}")
        st.metric(label="Davies-Bouldin Index", value=f"{db_score:.3f}")
    else:
        st.write("⚠️ No valid clusters were formed with the current parameters. Adjust `eps` or `min_samples`.")

    # Visualization
    st.write("### Clustering Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Cluster_DBSCAN'], palette='tab10', ax=ax)
    ax.set_title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
    st.pyplot(fig)

# Profiling Section
st.write("### 🔍 Cluster Profiling")
if "Cluster_KMeans" in df.columns:
    st.write("#### K-Means Profiling")
    profiling = df.groupby('Cluster_KMeans').agg({
        'Age': ['mean', 'std'],
        'Annual Income (k$)': ['mean', 'std'],
        'Spending Score (1-100)': ['mean', 'std']
    }).reset_index()
    st.dataframe(profiling)

if "Cluster_HC" in df.columns:
    st.write("#### Hierarchical Profiling")
    profiling = df.groupby('Cluster_HC').agg({
        'Age': ['mean', 'std'],
        'Annual Income (k$)': ['mean', 'std'],
        'Spending Score (1-100)': ['mean', 'std']
    }).reset_index()
    st.dataframe(profiling)

if "Cluster_DBSCAN" in df.columns:
    st.write("#### DBSCAN Profiling")
    profiling = df.groupby('Cluster_DBSCAN').agg({
        'Age': ['mean', 'std'],
        'Annual Income (k$)': ['mean', 'std'],
        'Spending Score (1-100)': ['mean', 'std']
    }).reset_index()
    st.dataframe(profiling)

# Download Results
st.write("### 📥 Download Cluster Assignments")
download_data = df[['CustomerID']]
if "Cluster_KMeans" in df.columns:
    download_data['Cluster_KMeans'] = df['Cluster_KMeans']
if "Cluster_HC" in df.columns:
    download_data['Cluster_HC'] = df['Cluster_HC']
if "Cluster_DBSCAN" in df.columns:
    download_data['Cluster_DBSCAN'] = df['Cluster_DBSCAN']

# CSV for Download
csv = download_data.to_csv(index=False)
st.download_button(label="Download Results as CSV", data=csv, file_name="cluster_results.csv", mime="text/csv")