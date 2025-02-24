import pickle
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px

# Step 1: Load the pickle file
file_path = r'C:/0github/Emotional-Data-Vids/Data/output_cleaned.pkl'  # Replace with your actual file path
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Step 2: Access the similarity matrix for a specific year or dataset (adjust as per your data)
df_2010 = data['similarity_pd_1014']  # Replace 'similarity_pd_1014' with the correct key in your pickle file

# Step 3: Apply t-SNE to reduce the dimensions to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
word_embeddings = tsne.fit_transform(df_2010.values)

# Step 4: Create a DataFrame with t-SNE results
words = df_2010.index  # Assuming the index contains the words
df_tsne = pd.DataFrame(word_embeddings, index=words, columns=['Dim1', 'Dim2'])

# Step 5: Cluster the words using KMeans clustering
n_clusters = 5  # You can adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_tsne['Cluster'] = kmeans.fit_predict(df_tsne[['Dim1', 'Dim2']])

# Step 6: Filter the words based on your criteria
filtered_words = ["DESIGN", "CIRCOLARE", "SOSTENIBILITà", "CREATIVITà", "MADE IN ITALY"]
df_filtered = df_tsne.loc[df_tsne.index.str.upper().isin(filtered_words)]

# Step 7: Plot the data using Plotly with color-coding for clusters
fig = px.scatter(
    df_tsne, 
    x='Dim1', y='Dim2', 
    color='Cluster',  # Color code by cluster
    hover_name=df_tsne.index,  # Show word on hover
    title="Word Similarity Visualization (t-SNE) with Clustering",
    labels={"Dim1": "Dimension 1", "Dim2": "Dimension 2"},
    width=800, height=600
)

# Highlight the filtered words
fig.add_scatter(
    x=df_filtered['Dim1'], 
    y=df_filtered['Dim2'], 
    mode='markers+text', 
    text=df_filtered.index, 
    textposition='top center',
    marker=dict(size=12, color='red', symbol='circle'),  # Different color and size for filtered words
    name='Filtered Words'
)

# Show the interactive plot
fig.show()
