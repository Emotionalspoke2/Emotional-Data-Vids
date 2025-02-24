import pickle
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px

# Carica il file pickle
file_path = r'C:/0github/Emotional-Data-Vids/Data/output_cleaned.pkl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Usa la matrice di similarità desiderata
similarity_df = data['similarity_pd_1014']

# Applica t-SNE
word_embeddings = TSNE(n_components=2, random_state=42).fit_transform(similarity_df.values)

# Crea il DataFrame dei risultati t-SNE
df_tsne = pd.DataFrame(word_embeddings, index=similarity_df.index, columns=['Dim1', 'Dim2'])

# Esegui KMeans clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_tsne['Cluster'] = kmeans.fit_predict(df_tsne[['Dim1', 'Dim2']])


cluster_da_visualizzare = 2  # cambia con il numero del cluster che vuoi isolare
df_cluster_isolato = df_tsne[df_tsne['Cluster'] == cluster_da_visualizzare]

# Visualizza solo questo cluster isolato
fig = px.scatter(
    df_cluster_isolato,
    x='Dim1', y='Dim2',
    hover_name=df_cluster_isolato.index,
    #color='Cluster',
    title=f"Visualizzazione del cluster {cluster_da_visualizzare} isolato (t-SNE)",
    labels={"Dim1": "Dimension 1", "Dim2": "Dimension 2"},
    width=800, height=600
)

fig.show()



