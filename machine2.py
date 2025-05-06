import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import folium
from folium.plugins import MarkerCluster
import random
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go

# Configurazione pagina
st.set_page_config(
    page_title="Localizzazione Negozio con K-Medoidi",
    page_icon="🏪",
    layout="wide"
)

# Titolo dell'app
st.title("🏪 Localizzazione Ottimale del Primo Negozio Fisico")
st.subheader("Dimostrazione dell'algoritmo K-Medoidi per l'e-commerce")

# Sidebar per i parametri
st.sidebar.header("Parametri")

# Parametri configurabili
num_clienti = st.sidebar.slider("Numero di clienti", 100, 2000, 1000)
num_medoids = st.sidebar.slider("Numero di negozi da aprire (k)", 1, 5, 1)
seed = st.sidebar.slider("Seed per la riproducibilità", 1, 100, 42)
peso_volume_acquisti = st.sidebar.slider(
    "Importanza del volume degli acquisti", 
    0.0, 2.0, 1.0, 
    help="0 = ignora volume acquisti, 1 = importanza normale, 2 = alta importanza"
)

# Definiamo alcune città italiane con le loro coordinate
@st.cache_data
def get_citta_italiane():
    return {
        "Milano": (45.4642, 9.1900),
        "Roma": (41.9028, 12.4964),
        "Napoli": (40.8518, 14.2681),
        "Torino": (45.0703, 7.6869),
        "Firenze": (43.7696, 11.2558),
        "Bologna": (44.4949, 11.3426),
        "Venezia": (45.4408, 12.3155),
        "Genova": (44.4056, 8.9463),
        "Bari": (41.1171, 16.8719),
        "Palermo": (38.1157, 13.3615)
    }

citta_italiane = get_citta_italiane()

# Distribuzione dei clienti (più clienti nelle città più grandi)
@st.cache_data
def get_pesi_citta():
    return {
        "Milano": 0.22,
        "Roma": 0.25,
        "Napoli": 0.12,
        "Torino": 0.08,
        "Firenze": 0.07,
        "Bologna": 0.07,
        "Venezia": 0.05,
        "Genova": 0.05,
        "Bari": 0.04,
        "Palermo": 0.05
    }

pesi_citta = get_pesi_citta()

# Personalizzazione della distribuzione dei clienti
st.sidebar.header("Distribuzione Clienti")
personalizza_distribuzione = st.sidebar.checkbox("Personalizza distribuzione clienti", False)

if personalizza_distribuzione:
    pesi_personalizzati = {}
    col1, col2 = st.sidebar.columns(2)
    for i, (citta, peso) in enumerate(pesi_citta.items()):
        if i % 2 == 0:
            pesi_personalizzati[citta] = col1.slider(f"{citta}", 0.01, 0.50, float(peso), 0.01)
        else:
            pesi_personalizzati[citta] = col2.slider(f"{citta}", 0.01, 0.50, float(peso), 0.01)
    
    # Normalizza i pesi per assicurarci che sommino a 1
    somma_pesi = sum(pesi_personalizzati.values())
    pesi_citta = {k: v/somma_pesi for k, v in pesi_personalizzati.items()}

# Implementazione dell'algoritmo K-Medoids (PAM - Partitioning Around Medoids)
def k_medoids(X, k, dist_matrix=None, max_iter=100):
    """
    Implementazione dell'algoritmo K-Medoids
    
    Parametri:
    - X: array di dati
    - k: numero di cluster
    - dist_matrix: matrice delle distanze (opzionale)
    - max_iter: numero massimo di iterazioni
    
    Restituisce:
    - medoids: indici dei medoidi
    - labels: etichette dei cluster per ogni punto
    - costs: costo totale
    """
    n_samples = X.shape[0]
    
    # Se non è fornita una matrice delle distanze, calcoliamola
    if dist_matrix is None:
        dist_matrix = pairwise_distances(X, metric='euclidean')
    
    # Inizializzazione: scegliamo k punti casuali come medoidi
    np.random.seed(seed)
    medoids = np.random.choice(n_samples, k, replace=False)
    
    # Inizializzazione etichette e costo
    labels = np.zeros(n_samples, dtype=int)
    best_cost = float('inf')
    
    for _ in range(max_iter):
        # Assegnazione: assegna ogni punto al medoide più vicino
        for i in range(n_samples):
            distances_to_medoids = [dist_matrix[i, medoid] for medoid in medoids]
            labels[i] = np.argmin(distances_to_medoids)
        
        # Aggiornamento: trova il miglior medoide per ogni cluster
        new_medoids = medoids.copy()
        for i in range(k):
            cluster_points = np.where(labels == i)[0]
            if len(cluster_points) > 0:
                # Calcola il costo totale per ogni potenziale medoide nel cluster
                costs = np.zeros(len(cluster_points))
                for j, point in enumerate(cluster_points):
                    costs[j] = sum(dist_matrix[point, other_point] for other_point in cluster_points)
                # Scegli il punto con il costo minimo come nuovo medoide
                new_medoids[i] = cluster_points[np.argmin(costs)]
        
        # Controlla se i medoidi sono cambiati
        if np.array_equal(medoids, new_medoids):
            break
        
        medoids = new_medoids
        
        # Calcola il costo totale
        current_cost = 0
        for i in range(n_samples):
            current_cost += dist_matrix[i, medoids[labels[i]]]
        
        if current_cost < best_cost:
            best_cost = current_cost
    
    return medoids, labels, best_cost

# Generazione dei dati
@st.cache_data
def genera_dati_clienti(num_clienti, pesi_citta, seed=42):
    """
    Genera un dataset sintetico di clienti con distribuzione geografica e volumi di acquisto.
    
    Parametri:
    - num_clienti: Numero totale di clienti da generare
    - pesi_citta: Dizionario con i pesi di distribuzione per ogni città
    - seed: Seed per garantire riproducibilità
    
    Restituisce:
    - DataFrame pandas con i dati dei clienti
    """
    # Impostiamo un seed per la riproducibilità
    np.random.seed(seed)
    random.seed(seed)
    
    clienti_lat = []
    clienti_lon = []
    citta_clienti = []
    date_prima_acquisto = []
    frequenza_acquisti = []
    categorie_preferite = []
    
    # Categorie di prodotti disponibili
    categorie = ['Abbigliamento', 'Elettronica', 'Casa', 'Bellezza', 
                 'Sport', 'Libri', 'Cibo', 'Giocattoli']
    
    # Date di acquisto in un intervallo di 2 anni precedenti
    data_inizio = pd.Timestamp('2023-01-01')
    data_fine = pd.Timestamp('2025-05-01')
    giorni_intervallo = (data_fine - data_inizio).days

    for _ in range(num_clienti):
        # Scegli una città in base ai pesi
        citta = random.choices(list(citta_italiane.keys()), 
                             weights=list(pesi_citta.values()), 
                             k=1)[0]
        citta_clienti.append(citta)
        
        # Ottieni coordinate di base
        lat_base, lon_base = citta_italiane[citta]
        
        # Aggiungi una piccola variazione casuale (per simulare indirizzi nella stessa città)
        # La deviazione standard è maggiore per le città più grandi
        std_dev = 0.03
        if citta in ['Roma', 'Milano']:
            std_dev = 0.05  # Più dispersione nelle grandi città
        elif citta in ['Napoli', 'Torino']:
            std_dev = 0.04  # Dispersione media nelle città medie
            
        lat = lat_base + np.random.normal(0, std_dev)
        lon = lon_base + np.random.normal(0, std_dev)
        
        clienti_lat.append(lat)
        clienti_lon.append(lon)
        
        # Genera data del primo acquisto casuale
        giorni_random = np.random.randint(0, giorni_intervallo)
        data_acquisto = data_inizio + pd.Timedelta(days=giorni_random)
        date_prima_acquisto.append(data_acquisto)
        
        # Genera frequenza di acquisto mensile (da 0.2 a 5 acquisti al mese)
        frequenza_acquisti.append(np.random.uniform(0.2, 5.0))
        
        # Genera categoria preferita
        categorie_preferite.append(random.choice(categorie))

    # Creiamo un DataFrame con i dati dei clienti
    df_clienti = pd.DataFrame({
        'lat': clienti_lat,
        'lon': clienti_lon,
        'citta': citta_clienti,
        'data_primo_acquisto': date_prima_acquisto,
        'frequenza_acquisti_mensile': frequenza_acquisti,
        'categoria_preferita': categorie_preferite
    })

    # Aggiungiamo una colonna con il volume di acquisti per cliente
    # Il volume dipende ora parzialmente dalla frequenza di acquisto
    base_volume = np.random.uniform(50, 500, len(df_clienti))
    df_clienti['volume_acquisti'] = base_volume * (0.7 + 0.3 * df_clienti['frequenza_acquisti_mensile'] / df_clienti['frequenza_acquisti_mensile'].max())
    
    # Aggiungiamo variabili demografiche sintetiche
    # Età dei clienti con distribuzione che dipende in parte dalla città
    eta_base = np.random.normal(35, 12, len(df_clienti))  # Età media 35, deviazione standard 12
    # Adattamento città-specifico
    for i, citta in enumerate(df_clienti['citta']):
        if citta in ['Milano', 'Roma', 'Bologna']:  # Città con più giovani
            eta_base[i] -= np.random.randint(0, 5)
        elif citta in ['Venezia', 'Firenze']:  # Città con età media più alta
            eta_base[i] += np.random.randint(0, 7)
    
    df_clienti['eta'] = np.clip(eta_base.astype(int), 18, 80)  # Limita età tra 18 e 80
    
    # Aggiungiamo il genere
    df_clienti['genere'] = np.random.choice(['M', 'F'], size=len(df_clienti))
    
    # Aggiungiamo fedeltà cliente (mesi di attività)
    oggi = pd.Timestamp('2025-05-01')
    df_clienti['mesi_attivita'] = ((oggi - df_clienti['data_primo_acquisto']).dt.days / 30).astype(int)
    
    # Calcola un punteggio di fedeltà combinando volume e mesi attività
    max_mesi = df_clienti['mesi_attivita'].max()
    if max_mesi > 0:  # Evita divisione per zero
        df_clienti['fedelta'] = (0.6 * df_clienti['volume_acquisti'] / df_clienti['volume_acquisti'].max() + 
                              0.4 * df_clienti['mesi_attivita'] / max_mesi) * 100
    else:
        df_clienti['fedelta'] = df_clienti['volume_acquisti'] / df_clienti['volume_acquisti'].max() * 100
    
    # Categorizzazione delle fasce di età
    conditions = [
        (df_clienti['eta'] < 25).values,
        ((df_clienti['eta'] >= 25) & (df_clienti['eta'] < 35)).values,
        ((df_clienti['eta'] >= 35) & (df_clienti['eta'] < 50)).values,
        ((df_clienti['eta'] >= 50) & (df_clienti['eta'] < 65)).values,
        (df_clienti['eta'] >= 65).values
    ]
    fascia_eta = ['18-24', '25-34', '35-49', '50-64', '65+']
    df_clienti['fascia_eta'] = np.select(conditions, fascia_eta, default='Unknown')
    
    # Aggiungiamo probabilità di acquisto in negozio fisico (basato su età e città)
    prob_base = np.random.uniform(0.3, 0.7, len(df_clienti))
    
    # Le persone più anziane preferiscono il negozio fisico
    for i, eta in enumerate(df_clienti['eta']):
        if eta > 50:
            prob_base[i] += np.random.uniform(0.1, 0.3)
        elif eta < 30:
            prob_base[i] -= np.random.uniform(0.05, 0.2)
    
    # In alcune città si preferisce lo shopping fisico
    for i, citta in enumerate(df_clienti['citta']):
        if citta in ['Milano', 'Roma', 'Napoli']:  # Città con forte cultura dello shopping
            prob_base[i] += np.random.uniform(0, 0.15)
    
    df_clienti['prob_acquisto_negozio'] = np.clip(prob_base, 0.1, 0.95)
    
    return df_clienti

# Genera i dati
df_clienti = genera_dati_clienti(num_clienti, pesi_citta, seed)

# Calcola la posizione ottimale del negozio
X = df_clienti[['lat', 'lon']].values

# Calcola la matrice delle distanze
dist_matrix = pairwise_distances(X, metric='euclidean')

# Applica il peso del volume degli acquisti
if peso_volume_acquisti > 0:
    for i in range(len(df_clienti)):
        dist_matrix[i, :] *= (1 / (df_clienti['volume_acquisti'].iloc[i] / df_clienti['volume_acquisti'].mean())) ** peso_volume_acquisti
        dist_matrix[:, i] *= (1 / (df_clienti['volume_acquisti'].iloc[i] / df_clienti['volume_acquisti'].mean())) ** peso_volume_acquisti

# Esegui l'algoritmo k-medoids
medoids, labels, cost = k_medoids(X, num_medoids, dist_matrix)

# Aggiungiamo l'etichetta del cluster al DataFrame
df_clienti['cluster'] = labels

# Layout principale - Mappa interattiva a tutta larghezza
st.header("Mappa Interattiva")

# Crea una mappa con folium
mappa = folium.Map(location=[42.8, 12.8], zoom_start=6)

# Crea un cluster per i marker dei clienti
marker_cluster = MarkerCluster().add_to(mappa)

# Colori per i cluster
colori_cluster = ['blue', 'green', 'purple', 'orange', 'darkred']

# Aggiungi i marker per i clienti
for idx, row in df_clienti.iterrows():
    cluster_id = int(row['cluster'])
    colore = colori_cluster[cluster_id % len(colori_cluster)]
    
    popup_text = f"""
    <b>Cliente {idx}</b><br>
    Città: {row['citta']}<br>
    Volume acquisti: €{row['volume_acquisti']:.2f}<br>
    Età: {row['eta']} anni<br>
    Genere: {row['genere']}<br>
    Categoria preferita: {row['categoria_preferita']}<br>
    Frequenza acquisti: {row['frequenza_acquisti_mensile']:.1f}/mese<br>
    Prob. acquisto in negozio: {row['prob_acquisto_negozio']:.2f}<br>
    Cluster: {cluster_id + 1}
    """
    
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=3,
        popup=popup_text,
        fill=True,
        fill_opacity=0.7,
        color=colore
    ).add_to(marker_cluster)

# Aggiungi i marker per i negozi (medoidi)
for i, medoid_idx in enumerate(medoids):
    negozio_lat = df_clienti['lat'].iloc[medoid_idx]
    negozio_lon = df_clienti['lon'].iloc[medoid_idx]
    negozio_citta = df_clienti['citta'].iloc[medoid_idx]
    
    popup_text = f"""
    <b>Negozio {i+1}</b><br>
    Città: {negozio_citta}<br>
    Coordinate: {negozio_lat:.4f}, {negozio_lon:.4f}
    """
    
    folium.Marker(
        location=[negozio_lat, negozio_lon],
        popup=popup_text,
        icon=folium.Icon(color='red', icon='shopping-cart', prefix='fa')
    ).add_to(mappa)

# Mostra la mappa in Streamlit
folium_static(mappa)

# Risultati dell'analisi e distribuzione clienti sotto la mappa
st.header("Risultati dell'Analisi")

# Layout per i risultati in due colonne
col1, col2 = st.columns([3, 2])

with col1:
    # Mostra la distribuzione dei clienti per città
    st.subheader("Distribuzione Clienti per Città")
    distribuzione_citta = df_clienti['citta'].value_counts().sort_values(ascending=False)
    
    fig = px.bar(
        x=distribuzione_citta.index,
        y=distribuzione_citta.values,
        labels={'x': 'Città', 'y': 'Numero di Clienti'},
        color=distribuzione_citta.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Mostra i risultati dei medoidi
    for i, medoid_idx in enumerate(medoids):
        negozio_lat = df_clienti['lat'].iloc[medoid_idx]
        negozio_lon = df_clienti['lon'].iloc[medoid_idx]
        negozio_citta = df_clienti['citta'].iloc[medoid_idx]
        
        st.subheader(f"Negozio {i+1}")
        st.write(f"📍 **Città ottimale**: {negozio_citta}")
        st.write(f"🌍 **Coordinate**: Lat {negozio_lat:.4f}, Lon {negozio_lon:.4f}")
        
        # Calcola il numero di clienti nel cluster
        num_clienti_cluster = (df_clienti['cluster'] == i).sum()
        percentuale_clienti = (num_clienti_cluster / len(df_clienti)) * 100
        
        st.write(f"👥 **Clienti serviti**: {num_clienti_cluster} ({percentuale_clienti:.1f}%)")

# Sezione inferiore
st.header("Visualizzazione dei Cluster")

# Aggiunta di informazioni sul clustering
st.markdown("""
### Interpretazione dei Cluster

I cluster rappresentano i gruppi di clienti che sarebbero serviti in modo ottimale da ciascun negozio fisico.
In questa visualizzazione:
- Ogni punto rappresenta un cliente
- La dimensione del punto indica il volume di acquisti
- Il colore indica il cluster di appartenenza
- Le stelle rosse rappresentano le posizioni ottimali dei negozi

Questa analisi permette di:
1. **Identificare bacini d'utenza** geograficamente coerenti
2. **Valutare la copertura territoriale** di ciascun potenziale negozio
3. **Stimare il potenziale commerciale** di ogni location in base ai clienti serviti
""")

# Grafico a dispersione con Plotly
fig = px.scatter(
    df_clienti, 
    x='lon', 
    y='lat', 
    color='cluster',
    size='volume_acquisti',
    size_max=15,
    hover_name='citta',
    hover_data=['volume_acquisti'],
    labels={'lon': 'Longitudine', 'lat': 'Latitudine', 'cluster': 'Cluster'},
    title='Clustering dei Clienti con K-Medoidi'
)

# Aggiungi i medoidi (negozi) al grafico
for i, medoid_idx in enumerate(medoids):
    fig.add_trace(go.Scatter(
        x=[df_clienti['lon'].iloc[medoid_idx]],
        y=[df_clienti['lat'].iloc[medoid_idx]],
        mode='markers',
        marker=dict(
            symbol='star',
            size=20,
            color='red',
            line=dict(width=2, color='DarkSlateGrey')
        ),
        name=f'Negozio {i+1}'
    ))

# Aggiungi etichette per le città principali
for citta, (lat, lon) in citta_italiane.items():
    fig.add_annotation(
        x=lon,
        y=lat,
        text=citta,
        showarrow=False,
        font=dict(size=10)
    )

st.plotly_chart(fig, use_container_width=True)

# Informazioni dettagliate sul dataset
with st.expander("Informazioni Dettagliate sul Dataset"):
    st.markdown(f"""
    ### Dataset Utilizzato
    
    Il dataset utilizzato in questa dimostrazione è generato in modo sintetico per simulare clienti di un e-commerce distribuiti nelle principali città italiane, con caratteristiche demografiche e comportamentali realistiche.
    
    #### Dimensioni e struttura del dataset:
    
    - **Numero totale di clienti**: {num_clienti} 
    - **Copertura geografica**: 10 principali città italiane
    - **Arco temporale**: Dati di acquisto dal 2023 a Maggio 2025
    
    #### Dati demografici:
    
    - **Età**: Distribuzione realistica tra 18-80 anni
      - Età media: {df_clienti['eta'].mean():.1f} anni
      - Distribuzione per fasce d'età:
    """)
    
    # Visualizza distribuzione fasce età
    fig_eta = px.histogram(
        df_clienti, 
        x='fascia_eta',
        category_orders={'fascia_eta': ['18-24', '25-34', '35-49', '50-64', '65+']},
        title='Distribuzione per Fasce d\'Età',
        color='fascia_eta',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_eta, use_container_width=True)
    
    # Continua descrizione dataset
    st.markdown(f"""
    - **Genere**: {(df_clienti['genere'] == 'M').sum()} maschi ({(df_clienti['genere'] == 'M').sum() / len(df_clienti) * 100:.1f}%) e {(df_clienti['genere'] == 'F').sum()} femmine ({(df_clienti['genere'] == 'F').sum() / len(df_clienti) * 100:.1f}%)
    
    #### Dati comportamentali:
    
    - **Volume acquisti**:
      - Range: €{df_clienti['volume_acquisti'].min():.2f} - €{df_clienti['volume_acquisti'].max():.2f}
      - Media: €{df_clienti['volume_acquisti'].mean():.2f}
      - Mediana: €{df_clienti['volume_acquisti'].median():.2f}
    
    - **Frequenza di acquisto mensile**:
      - Range: {df_clienti['frequenza_acquisti_mensile'].min():.1f} - {df_clienti['frequenza_acquisti_mensile'].max():.1f} acquisti/mese
      - Media: {df_clienti['frequenza_acquisti_mensile'].mean():.2f} acquisti/mese
    
    - **Anzianità cliente**:
      - Range: {df_clienti['mesi_attivita'].min()} - {df_clienti['mesi_attivita'].max()} mesi
      - Media: {df_clienti['mesi_attivita'].mean():.1f} mesi
    
    - **Categorie di prodotti preferite**:
    """)
    
    # Visualizza distribuzione categorie prodotti
    fig_cat = px.pie(
        df_clienti, 
        names='categoria_preferita',
        title='Distribuzione Categorie Prodotti Preferite',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_cat.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_cat, use_container_width=True)
    
    st.markdown(f"""
    - **Propensione all'acquisto in negozio fisico**:
      - Media: {df_clienti['prob_acquisto_negozio'].mean():.2f} (scala 0-1)
      - I clienti più anziani e quelli residenti in grandi città mostrano una maggiore propensione all'acquisto in negozio fisico
    
    #### Distribuzione geografica:
    
    - **Densità per città**:
      - Le maggiori concentrazioni di clienti si trovano a Roma ({(df_clienti['citta'] == 'Roma').sum()} clienti) e Milano ({(df_clienti['citta'] == 'Milano').sum()} clienti)
      - La distribuzione può essere personalizzata tramite i controlli nella barra laterale
    
    - **Variabilità geografica**:
      - I clienti sono distribuiti attorno alle coordinate centrali di ogni città con variazioni casuali
      - Le grandi città (Roma, Milano) presentano una maggiore dispersione geografica dei clienti
    
    #### Correlazioni notevoli:
    
    - Vi è una relazione positiva tra età e propensione all'acquisto in negozio fisico
    - I clienti con maggiore anzianità tendono ad avere volumi di acquisto più alti
    - La frequenza di acquisto e il volume totale sono correlati positivamente
    
    #### Rilevanza per il problema:
    
    Questo dataset simula una situazione realistica in cui un e-commerce desidera aprire negozi fisici nelle posizioni ottimali rispetto alla distribuzione dei propri clienti, considerando sia la loro posizione geografica che caratteristiche comportamentali come il valore commerciale, la fedeltà e la propensione all'acquisto in negozio.
    
    L'algoritmo k-medoidi è particolarmente adatto a questo problema perché identifica punti reali del dataset (clienti esistenti) come centri ottimali, fornendo quindi posizioni concrete dove localizzare i negozi.
    """)
    
    # Mostra un campione del dataframe
    st.subheader("Campione di dati (prime 10 righe)")
    st.dataframe(df_clienti.head(10))

# Aggiunta di una sezione informativa sull'applicazione pratica
st.header("Applicazione Pratica per l'E-commerce")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Vantaggi dell'approccio K-Medoidi")
    st.markdown("""
    - **Basato su dati reali**: La posizione ottimale corrisponde sempre a un cliente esistente, garantendo una location concreta e non teorica
    - **Considera il valore commerciale**: I clienti con volume di acquisti maggiore hanno più influenza sulla decisione
    - **Flessibilità**: Possibilità di valutare più location (aumentando k) per una strategia di espansione multi-negozio
    - **Robustezza**: Meno sensibile agli outlier rispetto ad altri algoritmi come K-Means
    """)

with col2:
    st.subheader("Fattori Decisionali Aggiuntivi")
    st.markdown("""
    Oltre ai risultati dell'algoritmo, un e-commerce dovrebbe considerare:
    
    - **Costi immobiliari** nelle location candidate
    - **Accessibilità e visibilità** del punto vendita
    - **Presenza di concorrenti** nelle vicinanze
    - **Potenziale di crescita** dell'area
    - **Caratteristiche demografiche** della zona
    
    Questa analisi fornisce una solida base quantitativa che dovrebbe essere integrata con valutazioni qualitative.
    """)
