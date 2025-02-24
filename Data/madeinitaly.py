import pickle
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px

# Caricamento file pickle
file_path = r'C:/0github/Emotional-Data-Vids/Data/output_cleaned.pkl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Funzione per eseguire TSNE e clustering, restituendo DataFrame pronto

def process_similarity_matrix(df_similarity, period_label, n_clusters=5):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(df_similarity.values)
    
    df_tsne = pd.DataFrame(embeddings, index=df_similarity.index, columns=['Dim1', 'Dim2'])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_tsne['Cluster'] = kmeans.fit_predict(df_tsne[['Dim1', 'Dim2']])
    df_tsne['Period'] = period_label

    return df_tsne

# Creazione DataFrames per periodi distinti
df_2010_2014 = process_similarity_matrix(data['similarity_pd_1014'], '2010-2014')
df_2015_2019 = process_similarity_matrix(data['similarity_pd_1519'], '2015-2019')
df_2020_2024 = process_similarity_matrix(data['similarity_pd_2024'], '2020-2024')

# Unione dei dataframe
combined_df = pd.concat([df_2010_2014, df_2015_2019, df_2020_2024])

# Parole filtrate da evidenziare
filtered_words = ["DESIGN", "CIRCOLARE", "SOSTENIBILITà", "CREATIVITà", "MADE IN ITALY"]
df_filtered = combined_df.loc[combined_df.index.str.upper().isin(filtered_words)]

# Grafico interattivo con Plotly
fig = px.scatter(
    combined_df,
    x='Dim1', y='Dim2',
    color='Cluster',
    facet_col='Period',  # crea sotto-grafici separati per ogni periodo temporale
    hover_name=combined_df.index,
    title="Word Similarity Visualization (t-SNE) by Period with Clustering",
    labels={"Dim1": "Dimension 1", "Dim2": "Dimension 2"},
    width=1200, height=600
)

# Evidenziazione parole filtrate
fig.add_scatter(
    x=df_filtered['Dim1'],
    y=df_filtered['Dim2'],
    mode='markers+text',
    text=df_filtered.index,
    textposition='top center',
    marker=dict(size=12, color='red', symbol='circle'),
    name='Filtered Words'
)

fig.show()
