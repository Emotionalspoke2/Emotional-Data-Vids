import pickle
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px

# Carica il file pickle
file_path = r'C:/0github/Emotional-Data-Vids/Data/output_cleaned.pkl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Definisci intervalli temporali e corrispondenti DataFrame
time_periods = {
    '2010-2014': data['similarity_pd_1014'],
    '2015-2019': data['similarity_pd_1519'],
    '2020-2024': data['similarity_pd_2024']
}

# Esegui t-SNE e clustering per ogni periodo temporale
results = {}
n_clusters = 5
cluster_da_visualizzare = 3

for period, df_similarity in time_periods.items():
    embeddings = TSNE(n_components=2, random_state=42).fit_transform(df_similarity.values)
    df_tsne = pd.DataFrame(embeddings, index=df_similarity.index, columns=['Dim1', 'Dim2'])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_tsne['Cluster'] = kmeans.fit_predict(df_tsne[['Dim1', 'Dim2']])
    df_tsne['Categoria'] = period
    results[period] = df_tsne[df_tsne['Cluster'] == cluster_da_visualizzare]

# Combina solo il cluster scelto per tutti i periodi
combined_df = pd.concat(results.values())

# Crea il grafico interattivo con bottoni per ciascun periodo temporale
fig = px.scatter(
    combined_df,
    x='Dim1', y='Dim2',
    hover_name=combined_df.index,
    color='Categoria',
    title=f"Visualizzazione del cluster Sostenibile isolato (t-SNE)",
    labels={"Dim1": "Dimension 1", "Dim2": "Dimension 2"},
    width=800, height=600
)

# Aggiorna layout con bottoni
fig.update_layout(
    updatemenus=[dict(
        buttons=list([
            dict(label="Tutti",
                 method="update",
                 args=[{"visible": [True, True, True]}, {"title": "Visualizzazione del cluster Sostenibile isolato (t-SNE)"}]),
            dict(label="2010-2014",
                 method="update",
                 args=[{"visible": [True, False, False]}, {"title": "Periodo 2010-2014"}]),
            dict(label="2015-2019",
                 method="update",
                 args=[{"visible": [False, True, False]}, {"title": "Periodo 2015-2019"}]),
            dict(label="2020-2024",
                 method="update",
                 args=[{"visible": [False, False, True]}, {"title": "Periodo 2020-2024"}])
        ]),
        direction="down"
    )]
)

fig.show()
fig.write_html("sostenibile.html")
