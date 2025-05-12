from IPython.display import SVG
from sknetwork.data import from_edge_list, from_adjacency_list, from_graphml, from_csv
from sknetwork.visualization import visualize_graph, visualize_bigraph
from sknetwork.clustering import Louvain, get_modularity
from networkx.algorithms.community import louvain_communities
from collections import Counter, defaultdict
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as c
from networkx.algorithms import community 
import numpy as np
import csv, unicodedata
from itertools import combinations
from pyvis.network import Network
from networkx import density
import random    
from itertools import combinations


def clean(name: str) -> str:
    n = unicodedata.normalize("NFKC", name.strip())
    n = " ".join(n.split())
    return n

def create_csv(name, dataset):
    """ Create a CSV file with co-authorship data.
     4 columns: author1, author2, weight, category """
    conditions = [
        dataset["category"].str.contains("stat\\.", case=False, na=False),
        dataset["category"].str.contains("math\\.", case=False, na=False),
        dataset["category"].str.contains("cs\\.",   case=False, na=False),
        dataset["category"].str.contains("econ\\.", case=False, na=False)
    ]
    choices = ["stat", "math", "cs", "econ"]
    dataset["category"] = np.select(conditions, choices, default=dataset["category"])

    coauthor_counts = defaultdict(int)

    for i, authors in enumerate(dataset["authors"]):
        category = dataset["category"].iat[i]
        authors_set = {clean(a) for a in authors.split(",") if a.strip()}
        for a1, a2 in combinations(sorted(authors_set), 2):
            coauthor_counts[(a1, a2, category)] += 1

    with open(name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["author1", "author2", "weight", "category"])
        for (a1, a2, cat), w in coauthor_counts.items():
            writer.writerow([a1, a2, w, cat])

    print(f"CSV sauvegardé sous « {name} »")


def nonDirected_graph(file, filter_weight=None, samples=None):
    ''' Create a Graph from
    file: collaboration csv with 4 columns: author1, author2, weight, category
    filter_weight: if we want to display only authors that have more than filter_weight collaborations
    samples: if the csv is too heavy, use only some samples taken randomly from the dataset
    '''
    df = pd.read_csv(file)

    # eliminate lines which weight isn't relevant
    if filter_weight is not None:
        df_filtered = df[df["weight"] > filter_weight]
        df = df_filtered
    
    # take randomly n samples
    if samples is not None:
        df_sample = df.sample(n=samples, random_state=42)  # random_state for reproducibility
        df = df_sample

    # Create non-directed graph
    G = nx.Graph()

    # Add weight and category as attributes to the edges
    for _, row in df.iterrows():
        G.add_edge(row["author1"], row["author2"], weight=row["weight"], category=row["category"])
        
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    return G


def distr_edges_weights(G):
    """
    Fonction pour afficher la distribution des poids des arêtes d'un graphe.
    """

    # 1. Récupérer tous les poids des arêtes
    weights = [edata['weight'] for _, _, edata in G.edges(data=True)]

    # 2. Compter combien de fois chaque poids apparaît
    weight_counts = Counter(weights)

    # 3. Trier les poids et leurs fréquences
    weights, counts = zip(*sorted(weight_counts.items()))

    # 4. Tracer l'histogramme
    plt.figure(figsize=(8, 5))
    plt.bar(weights, counts, log=True)  # Utiliser une échelle logarithmique pour l'axe Y
    plt.xlabel("Poids (weight)")
    plt.ylabel("Nombre d'arêtes (log scale)")
    plt.title("Distribution des poids des arêtes")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def distr_nodes_degree(G, w=False):
    """
    Affiche la distribution des degrés des noeuds du graphe G.
    w=False pour afficher la distribution des degrés (càd le nombre de voisins)
    w=True pour afficher la distribution des poids pondérés (càd le nombre d'articles publiés)
    """
    # Distribution des degrés des noeuds
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence

    # Créer une figure avec deux sous-graphiques côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if w==False:
        v = "(voisins)"
    else:
        v = "pondérés (nombre de publications)"

    # Histogramme avec échelle normale (à gauche)
    axes[0].hist(degree_sequence, bins=50, color='green', alpha=0.7)
    axes[0].axvline(np.mean(degree_sequence), color='red', linestyle='--', label=f'Moyenne: {np.mean(degree_sequence):.2f}')
    axes[0].set_xlabel("Degré (degree)")
    axes[0].set_ylabel("Nombre de noeuds (nodes)")
    axes[0].set_title("Distribution des degrés, ",v," des noeuds (échelle normale)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Histogramme avec échelle logarithmique (à droite)
    axes[1].hist(degree_sequence, bins=50, color='green', alpha=0.7, log=True)  # Log scale for y-axis
    axes[1].axvline(np.mean(degree_sequence), color='red', linestyle='--', label=f'Moyenne: {np.mean(degree_sequence):.2f}')
    axes[1].axvline(np.median(degree_sequence), color='purple', linestyle='--', label=f'Médiane: {np.median(degree_sequence):.2f}')
    axes[1].set_xscale('log')  # Log scale for x-axis
    axes[1].set_xlabel("Degré (degree) [log scale]")
    axes[1].set_ylabel("Nombre de noeuds (nodes) [log scale]")
    axes[1].set_title("Distribution des degrés ", v ," (voisins) des noeuds (échelle logarithmique)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    # Ajuster l'espacement entre les sous-graphiques
    plt.tight_layout()
    plt.show()

def add_node_attribut_primary_cat(G):
    """ Ajouter l'attribut 'primary_category' à chaque noeud du graphe G"""
    primary_cat = {}
    for node in G:
        cats = [edata['category'] for _, _, edata in G.edges(node, data=True)]
        if cats:
            primary_cat[node] = Counter(cats).most_common(1)[0][0] # example: primary_cat[node] = 'cs'
        else:
            primary_cat[node] = None
    nx.set_node_attributes(G, primary_cat, 'primary_category')
    return primary_cat

def edge_max(G, primary_cat):
    """Trouver l'arête avec le poids maximal"""
    max_weight = max([edata['weight'] for _, _, edata in G.edges(data=True)])
    for u, v, edata in G.edges(data=True):
        if edata['weight'] == max_weight:
            print(f"Les deux auteurs connectés par l'arête de poids maximal ({max_weight}) sont : '{u}' et '{v}'")
            # leur degré
            print(f"Leur degré, càd le nombre de publications est : {G.degree(u, weight='weight')}, {G.degree(v, weight='weight')}")
            print("leur catégorie principale:")
            print(primary_cat[u], primary_cat[v])
            break

def louvain_attribut(G, res=1.0):
    """ Run louvain algorith
    communities c'est une liste d'ensembles, chaque ensemble est composé des auteurs de la communauté: [{"john", "clara"}, {"jane",...}, ...]
    membership c'est un dictionnaire qui associe chaque auteur à sa communauté: {"john" : 0, "clara" : 0, "jane": 1, ...}
    ajouter un attribut à chaque noeud du graphe: la communauté à laquelle il appartient"""
    communities = louvain_communities(G, weight='weight', resolution=res) 

    membership = {node: cid for cid, comm in enumerate(communities) for node in comm}
    nx.set_node_attributes(G, membership, 'louvain_community')

    #Print the number of communities
    print(f"Number of communities: {len(communities)}")

    return communities, membership

def visualize_with_pyvis(G, output_file='network.html'):
    """
    Create an interactive visualization of the network using pyvis
    
    Parameters:
    G (nx.Graph): NetworkX graph object
    output_file (str): Name of the HTML file to save the visualization
    """
    # Create a pyvis network
    net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='remote')
    
    # Get weights for edge scaling
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(weights) if weights else 1
    
    # Add nodes and edges to the pyvis network
    for node in G.nodes():
        net.add_node(node, size=10, title=node)
    
    # 3. Add edges with exact weights as widths
    for u, v in G.edges():
        weight = G[u][v]['weight']
        # Scale width (1-10) based on weight
        #width = 1 + 9 * (weight / max_weight)
        net.add_edge(u, v, value=weight, width=weight, title=f"Collaborations: {weight}")
    
    
    # Scale node size by degree:
    degrees = dict(G.degree())
    for node in net.nodes:
        node['size'] = 5 + degrees[node['id']]  # Scale size by degree

    # Color nodes by community:
    communities = community.greedy_modularity_communities(G)
    for i, com in enumerate(communities):
        for node in com:
            net.get_node(node)['color'] = f'hsl({i*60}, 100%, 50%)'

    # Configure physics for better layout
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -80000,
          "centralGravity": 0.3,
          "springLength": 200,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.1
        },
        "minVelocity": 0.75
      }
    }
    """)
    
    # Save and show the network
    net.show(output_file)
    return net


def density_hist_comm_size(membership):
    """histogramme de densité des tailles des communautés"""
    community_sizes = Counter(membership.values())
    plt.figure(figsize=(10, 6))
    plt.hist(list(community_sizes.values()), bins=150,  color='skyblue', edgecolor='black')
    plt.xlabel('Taille de la communauté')
    plt.ylabel('Nombre de communnautés')
    plt.title('Distribution de la taille des communautés')
    plt.yscale('log')  # Use log scale for better visualization if there are large differences
    plt.show()



def are_little_comms_dense(G,communities,membership, t):
    """Vérifier la densité des petites communautés (<10 nœuds)
    Densité = 2m/n(n-1) où m est le nombre d'arêtes et n le nombre de nœuds"""
    small_communities = [comm for comm in communities if len(comm) < 10]

    # Seuil pour définir une communauté dense (modifiable)
    t = 0.99
    dens=0
    results = []
    for comm in small_communities:
        subg = G.subgraph(comm)
        d = density(subg)
        # Récupère l'ID de la communauté à partir d'un de ses nœuds
        cid = membership[next(iter(comm))]
        results.append((cid, len(comm), d, d > t))
        if d> t:
            dens+=1

    # Calculer la fréquence (probabilité) d'être dense parmi les petites communautés
    freq_dense = dens / len(small_communities) if small_communities else 0
    print(f"Nombre de petites communautés (<10): {len(small_communities)}")
    print(f"Nombre de communautés denses (densité > {t}): {dens}")
    print(f"Probabilité (fréquence) d'être dense : {freq_dense:.2%}")
    print(f"threshold: {t}")

    # Afficher un camembert pour visualiser la proportion
    plt.figure(figsize=(5, 5))
    plt.pie([dens, len(small_communities) - dens],
        labels=['Denses', 'Non denses'],
        autopct='%1.1f%%',
        colors=['#66c2a5', '#fc8d62'])
    plt.title(f"Proportion de petites communautés denses (<10 nœuds)")
    plt.show()

def are_big_comms_dense(G,communities,membership, t):
    """Vérifier la densité des petites communautés (<10 nœuds)
    Densité = 2m/n(n-1) où m est le nombre d'arêtes et n le nombre de nœuds"""
    big_communities = [comm for comm in communities if len(comm) > 100]

    # Seuil pour définir une communauté dense (modifiable)
    t = 0.99
    dens=0
    results = []
    for comm in big_communities:
        subg = G.subgraph(comm)
        d = density(subg)
        # Récupère l'ID de la communauté à partir d'un de ses nœuds
        cid = membership[next(iter(comm))]
        results.append((cid, len(comm), d, d > t))
        if d> t:
            dens+=1

    # Calculer la fréquence (probabilité) d'être dense parmi les petites communautés
    freq_dense = dens / len(big_communities) if big_communities else 0
    print(f"Nombre de petites communautés (<10): {len(big_communities)}")
    print(f"Nombre de communautés denses (densité > {t}): {dens}")
    print(f"Probabilité (fréquence) d'être dense : {freq_dense:.2%}")
    print(f"threshold: {t}")

    # Afficher un camembert pour visualiser la proportion
    plt.figure(figsize=(5, 5))
    plt.pie([dens, len(big_communities) - dens],
        labels=['Denses', 'Non denses'],
        autopct='%1.1f%%',
        colors=['#66c2a5', '#fc8d62'])
    
    plt.title(f"Proportion de grandes communautés denses (>100 nœuds)")
    plt.show()

def mean_edges_weight(G, communities, size, little=False):
    """Calculer le poids moyen des arêtes des communautés de taille > size si little =False et de taille < size si little = True"""

    community_means = []
    community_ids = []

    for cid, comm in enumerate(communities):
        if little:
            if len(comm) < size:
                subg = G.subgraph(comm)
                weights = [edata['weight'] for _, _, edata in subg.edges(data=True)]
                mean_weight = np.mean(weights) if weights else 0
                community_means.append(mean_weight)
                community_ids.append(cid)
            v = "<"
        else:
            if len(comm) > size:
                subg = G.subgraph(comm)
                weights = [edata['weight'] for _, _, edata in subg.edges(data=True)]
                mean_weight = np.mean(weights) if weights else 0
                community_means.append(mean_weight)
                community_ids.append(cid)
            v = ">"

    plt.figure(figsize=(10, 5))
    plt.scatter(community_ids, community_means, alpha=0.7)
    plt.xlabel("ID de la communauté")
    plt.ylabel("Poids moyen des arêtes")
    plt.title("Poids moyen des arêtes pour les communautés (",v,size,")")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Ajout de la moyenne sur l'axe des y
    mean_y = np.mean(community_means)
    plt.axhline(mean_y, color='red', linestyle='--', label=f'Moyenne: {mean_y:.2f}')
    plt.axhline(np.median(community_means), color='green', linestyle='--', label=f'Médiane: {np.median(community_means):.2f}')
    plt.legend()

    plt.show()

def first_analysis_giant_connected_component(G):

    """ return: Gg un sous-graphe du plus grand composant connexe de G
    La proportion des noeuds de G qui composent le plus grand composant connexe
    La moyenne des poids des arêtes de ce composant"""

    giant = max(nx.connected_components(G), key=len)
    Gg = G.subgraph(giant)
    relative_size = Gg.number_of_nodes() / G.number_of_nodes()
    print(f"Taille relative du plus grand composant connexe :{Gg.number_of_nodes()} / {G.number_of_nodes()} = {relative_size:.4f}")
    mean_weight_giant = np.mean([edata['weight'] for _, _, edata in Gg.edges(data=True)])
    print(f"Moyenne des poids des arêtes de la composante géante : {mean_weight_giant:.4f}")
    return Gg

def plyvalent_authors_by_category(G, primary_cat):
    """ Trouver les auteurs qui sont des ponts entre les catégories
    (càd qui ont le plus d'arêtes avec des catégories différentes de la leur)"""
    bridge_score = {}
    for node in G:
        pc = primary_cat[node]
        # count edges whose category ≠ this node’s primary_category
        cross = sum(1 for _, _, ed in G.edges(node, data=True)
                    if ed['category'] != pc)
        bridge_score[node] = cross
    best_bridges = defaultdict(lambda: (None, -1))

    for node, score in bridge_score.items():
        pc = primary_cat[node]
        if node==' Jr.':
            continue 
        # this author’s score bigger than the current stored best score for that category
        if pc and score > best_bridges[pc][1]:
            best_bridges[pc] = (node, score)

    # Show results
    for cat, (author, score) in best_bridges.items():
        print(f"Category '{cat}': bridge author = {author!r} with {score} cross‐category edges and in community {G.nodes[author]['louvain_community']}")


def bridge_authors_by_centrality(G, primary_cat):
    cats = sorted({c for c in primary_cat.values() if c is not None})

    all_nodes = list(G.nodes())
    sample_nodes = set(random.sample(all_nodes, 900000))

    # Création du sous‐graphe induit (et copie pour avoir un graphe indépendant)
    G_sub = G.subgraph(sample_nodes)
    # PERSONNES-PONT entre chaque paire (A→B) : 
    # auteurs de A avec le PLUS de voisins dans B et un betweenness élevé.
    betw = nx.betweenness_centrality(G_sub, weight='weight')

    bridges = {}
    for c1, c2 in combinations(cats, 2):
        # candidats dans c1 → voisins dans c2
        scores = []
        for u in G_sub.nodes:
            if primary_cat.get(u) != c1:
                continue
            nbrs_in_c2 = sum(1 for v in G_sub[u] if primary_cat.get(v)==c2)
            if nbrs_in_c2>0:
                scores.append((u, nbrs_in_c2, betw[u]))
        # Triez d’abord par voisins (desc), puis par betw (desc)
        scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        if scores:
            u, nbrs, b = scores[0]
            bridges[(c1,c2)] = (u, nbrs, b)

    print(">>> Personnes-pont (A→B) : (auteur, #voisins_B, betweenness)")
    for (c1,c2), (u,nv,b) in bridges.items():
        print(f"  {c1:>6}→{c2:<6} : {u!r}, voisins={nv}, betw={b:.4f}")


def leaders_by_category(G, primary_cat, degree=True, betw=False ):
    """ Cherche des leaders par catégorie
        degree=True : extrait le sous-graphe de chaque catégorie et prend le nœud de degré pondéré maximal.
        betw=True: Betweenness : même chose mais avec la mesure de betweenness sur chaque sous-graphe."""

    cats = sorted({c for c in primary_cat.values() if c is not None})

    if degree:
        # LEADERS par catégorie par centralité de degré (pondéré)
        leaders_deg = {}
        for c in cats:
            sub = G.subgraph([u for u in G if primary_cat.get(u)==c])
            # degree pondéré
            deg = sub.degree(weight='weight')
            u, d = max(deg, key=lambda x: x[1])
            leaders_deg[c] = (u, d)

        print("\n>>> Leaders par degré (catégorie: auteur, degré_pondéré)")
        for c, (u,d) in leaders_deg.items():
            print(f"  {c:>8} : {u!r}, degré={d}")
    else:
        leaders_deg = None
    if betw==True:
        # Leaders par catégorie par betweenness dans chaque sous-graphe
        leaders_btw = {}
        for c in cats:
            sub = G.subgraph([u for u in G if primary_cat.get(u)==c])
            btw_sub = nx.betweenness_centrality(sub, weight='weight')
            u, b = max(btw_sub.items(), key=lambda x: x[1])
            leaders_btw[c] = (u, b)

        print("\n>>> Leaders par betweenness (catégorie: auteur, betweenness)")
        for c, (u,b) in leaders_btw.items():
            print(f"  {c:>8} : {u!r}, betw={b:.4f}")
    else:
        leaders_deg = None
    return leaders_deg


def analyze_distance_distributions(G, primary_cat, leaders_deg, n_samples=1000):
    """
    Pour G (avec node attr primary_category) :
      1) Distance random_intra : pour chaque catégorie, on tire n_samples paires
         (u,v) dans la même catégorie, calcule leur distance, trace et moyenne.
      2) Distance random_any : on tire n_samples paires de nœuds au hasard dans G,
         calcule leur distance, trace et moyenne.
    leaders_deg doit être un dict {cat: (leader_node, deg)}.
    """
    # Préparation : on travaillera sur le plus grand composant pour éviter les isolés
    comp0 = max(nx.connected_components(G), key=len)
    G0 = G.subgraph(comp0).copy()
    nodes0 = list(G0.nodes())
    
    # 1) random intra-cat
    fig, axes = plt.subplots(1, len(leaders_deg), figsize=(4*len(leaders_deg), 4))
    if len(leaders_deg)==1: axes = [axes]
    for ax, cat in zip(axes, leaders_deg):
        authors = [u for u in G0 if primary_cat.get(u)==cat]
        pairs = [random.sample(authors, 2) for _ in range(n_samples)]
        dists = []
        for u,v in pairs:
            try:
                dists.append(nx.shortest_path_length(G0, u, v))
            except nx.NetworkXNoPath:
                pass
        ax.hist(dists, bins=20)
        ax.axvline(sum(dists)/len(dists), color='k', linestyle='--')
        ax.set_title(f"Random intra {cat}\nmean={sum(dists)/len(dists):.2f}")
        ax.set_xlabel('Distance')
        ax.set_ylabel('Count')
    plt.tight_layout()
    plt.show()

    # 2) random any
    pairs = [random.sample(nodes0, 2) for _ in range(n_samples)]
    dists = []
    for u,v in pairs:
        try:
            dists.append(nx.shortest_path_length(G0, u, v))
        except nx.NetworkXNoPath:
            pass
    plt.figure(figsize=(5,4))
    plt.hist(dists, bins=20)
    plt.axvline(sum(dists)/len(dists), color='k', linestyle='--')
    plt.title(f"Random any-category\nmean={sum(dists)/len(dists):.2f}")
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def run_random_separation(G, runs=10, max_deg=8, seed=None, plot=True):
    """
    Pour `runs` nœuds aléatoires dans le grand composant de G, calcule la taille
    cumulée de l'ensemble atteint à chaque degré de séparation (1..max_deg).
    Si plot=True, trace les `runs` courbes cumulatives pour comparer les profils.
    """
    if seed is not None:
        random.seed(seed)
    # grand composant
    comp0 = max(nx.connected_components(G), key=len)
    G0 = G.subgraph(comp0)
    nodes = list(G0.nodes())
    # tirage aléatoire de `runs` nœuds dans le grand composant
    starts = random.sample(nodes, runs)
    
    profiles = {}
    for s in starts:
        visited = {s}
        frontier = {s}
        cum_counts = []
        for d in range(1, max_deg + 1):
            #voisins de frontier
            neigh = set(nb for u in frontier for nb in G0.neighbors(u))
            # ne garder que les nouveaux
            new_front = neigh - visited
            visited |= new_front
            cum_counts.append(len(visited))
            frontier = new_front
            if not frontier:
                break
        profiles[s] = cum_counts

    if plot:
        plt.figure()
        for s, cum in profiles.items():
            plt.plot(range(1, len(cum) + 1), cum, marker='o', label=str(s))
        plt.xlabel('Degré de séparation')
        plt.ylabel('Nombre cumulatif d’auteurs atteints')
        plt.title('Profils de séparation pour 10 démarrages aléatoires')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return profiles


def transitivity_by_category(G, primary_cat):
    """ TRANSITIVITÉ (global clustering) par catégorie"""
    cats = sorted({c for c in primary_cat.values() if c is not None})
    trans = {}
    for c in cats:
        sub = G.subgraph([u for u in G if primary_cat.get(u)==c])
        trans[c] = nx.transitivity(sub)

    print("\n>>> Transitivité (2m/[n(n-1)]) par catégorie")
    for c, t in trans.items():
        print(f"  {c:>8} : {t:.4f}")