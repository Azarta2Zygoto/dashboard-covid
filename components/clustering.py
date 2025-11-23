import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings


class COVIDClustering1:
    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
    
      
    def prepare_data(self, features, date=None, normalize=True):
        """Prétraitement des données pour le clustering."""
        if date:
            data = self.df[self.df['date'] == date].copy()
        else:
            data = self.df.sort_values('date').groupby('location').last().reset_index()
        
        base_columns = ['location', 'iso_code', 'continent']
        available_features = [f for f in features if f in data.columns]
        
        clustering_data = data[base_columns + available_features].copy()

        # Gestion des valeurs manquantes
        clustering_data = clustering_data.dropna(subset=available_features)

        if normalize:
            X = clustering_data[available_features].copy()
            X_imputed = self.imputer.fit_transform(X)
            X_normalized = self.scaler.fit_transform(X_imputed)
            clustering_data[available_features] = X_normalized

        print(f" Données prétraitées: {len(clustering_data)} pays, {len(available_features)} features")
        return clustering_data, available_features
    
    def find_optimal_k_silhouette(self, X, max_k=8):
        """Trouver le bon k avec la méthode du silhouette score"""
        if len(X) < 3:
            return 2, []
        
        k_range = range(2, min(max_k + 1, len(X)))
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            if len(np.unique(labels)) < 2:
                silhouette_scores.append(-1)
                continue
                
            silhouette_avg = silhouette_score(X, labels)
            silhouette_scores.append(silhouette_avg)
        if not silhouette_scores:
            return 2, []
            
        best_k = k_range[np.argmax(silhouette_scores)]
        return best_k, silhouette_scores
    
    def find_optimal_k_elbow(self, X, max_k=8):
        """Trouve le k optimal avec la méthode du coude ou elbow """
        if len(X) < 3:
            return 2, []
        
        k_range = range(1, min(max_k + 1, len(X)))
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        # Détection du "coude" - où la réduction d'inertie ralentit
        if len(inertias) > 2:
            differences = []
            for i in range(1, len(inertias)):
                diff = inertias[i-1] - inertias[i]
                differences.append(diff)
            
            # Trouver où la réduction diminue significativement
            best_k = 2
            for i in range(1, len(differences)):
                if differences[i] < differences[i-1] * 0.6:
                    best_k = i + 1
                    break
        else:
            best_k = 2
        
        return best_k, inertias
    
    def perform_kmeans(self, features, n_clusters=None, date=None, auto_select=True):
        """K-means avec sélection automatique du k optimal (Auto K)"""
        clustering_data, available_features = self.prepare_data(features, date, normalize=True)
        X = clustering_data[available_features].values
        
        if auto_select and n_clusters is None:
            n_clusters, _ = self.find_optimal_k_silhouette(X)
            print(f"K optimal (silhouette): {n_clusters}")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
        labels = kmeans.fit_predict(X)
        clustering_data['cluster'] = labels
        clustering_data['cluster'] = clustering_data['cluster'].astype(str)
        
        # Calcul du score de silhouette
        if len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(X, labels)
        else:
            silhouette_avg = -1
        
        return clustering_data, kmeans, available_features, silhouette_avg
    
    def perform_gaussian_mixture(self, features, n_components=None, date=None):
        """Gaussian Mixture """
        clustering_data, available_features = self.prepare_data(features, date, normalize=True)
        X = clustering_data[available_features].values
        
        if n_components is None:
            n_components, _ = self.find_optimal_k_silhouette(X)
        
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        labels = gmm.fit_predict(X)
        
        clustering_data['cluster'] = labels
        clustering_data['cluster'] = clustering_data['cluster'].astype(str)
        
        return clustering_data, gmm, available_features
    
    def perform_dbscan(self, features, eps=0.5, min_samples=3, date=None):
        """DBSCAN pour clusters de forme arbitraire comme on a vu en cours"""
        clustering_data, available_features = self.prepare_data(features, date, normalize=True)
        X = clustering_data[available_features].values
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        clustering_data['cluster'] = labels
        clustering_data['cluster'] = clustering_data['cluster'].astype(str)
        
        # DBSCAN peut créer du bruit (cluster -1)
        n_clusters = len(np.unique(labels[labels != -1]))
        print(f"DBSCAN: {n_clusters} clusters + bruit")
        
        return clustering_data, dbscan, available_features
    
    def compare_methods(self, features, date=None):
        """Compare les méthodes """
        results = {}
        
        # K-means
        try:
            kmeans_data, kmeans_model, kmeans_features, silhouette = self.perform_kmeans(features, date=date)
            results['kmeans'] = {
                'data': kmeans_data,
                'model': kmeans_model,
                'silhouette': silhouette,
                'n_clusters': kmeans_data['cluster'].nunique()
            }
        except Exception as e:
            print(f"K-means failed: {e}")

        # Gaussian Mixture
        try:
            gmm_data, gmm_model, gmm_features = self.perform_gaussian_mixture(features, date=date)
            results['gmm'] = {
                'data': gmm_data,
                'model': gmm_model,
                'n_clusters': gmm_data['cluster'].nunique()
            }
        except Exception as e:
            print(f"Gaussian mixture failed: {e}")

        # DBSCAN
        try:
            dbscan_data, dbscan_model, dbscan_features = self.perform_dbscan(features, date=date)
            results['dbscan'] = {
                'data': dbscan_data,
                'model': dbscan_model,
                'n_clusters': dbscan_data[dbscan_data['cluster'] != '-1']['cluster'].nunique()
            }
        except Exception as e:
            print(f" DBSCAN failed: {e}")
        
        return results
    
    def _generate_cluster_label(self, high_features, low_features, size):
        """Génère un label descriptif pour les clusters"""
    
        # Mapping des features avec des descriptions plus significatives
        feature_descriptions = {
            'total_cases_per_million': 'taux de cas',
            'total_deaths_per_million': 'taux de mortalité',
            'gdp_per_capita': 'richesse économique',
            'human_development_index': 'niveau de développement',
            'aged_65_older': 'population âgée',
            'population_density': 'densité de population',
            'people_fully_vaccinated_per_hundred': 'taux de vaccination',
            'new_cases_per_million': 'nouveaux cas',
            'new_deaths_per_million': 'nouvelles morts',
            'reproduction_rate': 'taux de reproduction',
            'positive_rate': 'taux de positivité',
            'tests_per_case': 'tests par cas'
        }
    
        # Catégorisation des caractéristiques
        economic_indicators = ['gdp_per_capita', 'human_development_index']
        health_indicators = ['total_cases_per_million', 'total_deaths_per_million', 'new_cases_per_million', 'new_deaths_per_million']
        demographic_indicators = ['aged_65_older', 'population_density']
        testing_indicators = ['positive_rate', 'tests_per_case']
        vaccination_indicators = ['people_fully_vaccinated_per_hundred']
        
        # Traduire les features en descriptions françaises
        def translate_features(feature_list):
            return [feature_descriptions.get(f, f) for f in feature_list]
        
        high_translated = translate_features(high_features)
        low_translated = translate_features(low_features)
        
        # Déterminer le profil général du cluster
        profile_characteristics = []
        
        # Analyser les indicateurs économiques
        economic_high = [f for f in high_features if f in economic_indicators]
        economic_low = [f for f in low_features if f in economic_indicators]
        
        if economic_high and not economic_low:
            profile_characteristics.append("économiquement développés")
        elif economic_low and not economic_high:
            profile_characteristics.append("économiquement en développement")
        
        # Analyser les indicateurs de santé
        health_high = [f for f in high_features if f in health_indicators]
        health_low = [f for f in low_features if f in health_indicators]
        
        if 'taux de mortalité' in high_translated or 'nouvelles morts' in high_translated:
            profile_characteristics.append("fort impact sanitaire")
        elif 'taux de mortalité' in low_translated or 'nouvelles morts' in low_translated:
            profile_characteristics.append("faible impact sanitaire")
        
        if 'taux de cas' in high_translated or 'nouveaux cas' in high_translated:
            profile_characteristics.append("forte propagation")
        
        # Analyser la démographie
        if 'population âgée' in high_translated:
            profile_characteristics.append("population vieillissante")
        if 'densité de population' in high_translated:
            profile_characteristics.append("forte densité")
        
        # Analyser la vaccination
        if 'taux de vaccination' in high_translated:
            profile_characteristics.append("bonne couverture vaccinale")
        elif 'taux de vaccination' in low_translated:
            profile_characteristics.append("faible couverture vaccinale")
    
    # Construire le label final
        if profile_characteristics:
            profile = ", ".join(profile_characteristics)
            return f"Pays {profile} ({size} pays)"
        else:
            # Si y'a pas de profil clair, utiliser les caractéristiques principales
            main_features = []
            if high_translated:
                main_features.extend([f"fort {f}" for f in high_translated[:2]])
            if low_translated:
                main_features.extend([f"faible {f}" for f in low_translated[:2]])
            
            if main_features:
                features_str = ", ".join(main_features)
                return f"Pays à {features_str} ({size} pays)"
            else:
                return f"Cluster moyen ({size} pays)"

    def interpret_clusters(self, clustering_data, features):
        """Interprète les clusters avec des descriptions plus significatives"""
        interpretations = []
        
        for cluster in sorted(clustering_data['cluster'].unique()):
            cluster_data = clustering_data[clustering_data['cluster'] == cluster]
            
            # Calcul des centroïdes (moyennes)
            centroid = cluster_data[features].mean()
            global_mean = clustering_data[features].mean()
            
            # Features distinctives avec seuils ajustés
            distinctive_high = []
            distinctive_low = []

            for feature in features:
                ratio = centroid[feature] / global_mean[feature]
                # Seuils plus stricts pour éviter les faux positifs
                if ratio > 1.3:  # Augmenté de 1.2 à 1.3
                    distinctive_high.append(feature)
                elif ratio < 0.7:  # Diminué de 0.8 à 0.7
                    distinctive_low.append(feature)
            
            # Trier par importance (écart le plus grand)
            distinctive_high.sort(key=lambda f: abs(centroid[f] / global_mean[f] - 1), reverse=True)
            distinctive_low.sort(key=lambda f: abs(1 - centroid[f] / global_mean[f]), reverse=True)
            
            # Génération de label amélioré
            label = self._generate_cluster_label(distinctive_high, distinctive_low, len(cluster_data))

            # Pays représentatifs (essayer d'avoir une diversité géographique)
            countries_list = cluster_data['location'].head(5).tolist()
            
            # Statistiques clés pour l'interprétation
            key_stats = {}
            for feature in ['total_cases_per_million', 'total_deaths_per_million', 
                        'gdp_per_capita', 'human_development_index']:
                if feature in features:
                    key_stats[feature] = {
                        'cluster_mean': centroid[feature],
                        'global_mean': global_mean[feature],
                        'ratio': centroid[feature] / global_mean[feature]
                    }

            interpretations.append({
                'cluster': cluster,
                'size': len(cluster_data),
                'label': label,
                'countries': countries_list,
                'distinctive_high': distinctive_high[:3],
                'distinctive_low': distinctive_low[:3],
                'key_stats': key_stats,
                'centroid': centroid.to_dict()
            })
        
        return interpretations
    
    def create_silhouette_analysis(self, clustering_data, features):
        """Analyse silhouette pour évaluer la qualité du clustering."""
        X = clustering_data[features].values
        labels = clustering_data['cluster'].astype(int).values
        
        if len(np.unique(labels)) < 2:
            return None
        
        silhouette_avg = silhouette_score(X, labels)
        sample_silhouette_values = silhouette_samples(X, labels)
        
        clustering_data['silhouette_score'] = sample_silhouette_values
        
        # Visualisation silhouette
        fig = go.Figure()
        
        for cluster in np.unique(labels):
            cluster_silhouette_vals = sample_silhouette_values[labels == cluster]
            cluster_silhouette_vals_sorted = np.sort(cluster_silhouette_vals)
            
            y = np.arange(len(cluster_silhouette_vals))
            fig.add_trace(go.Scatter(
                    x=cluster_silhouette_vals_sorted,
                    y=y,
                    mode='lines',
                    name=f'Cluster {cluster}',
                    fill='tozerox'
                ))
        
        fig.add_vline(x=silhouette_avg, line_dash="dash", line_color="red",
                     annotation_text=f"Score moyen: {silhouette_avg:.3f}")
        
        fig.update_layout(
            title="Analyse Silhouette",
            xaxis_title="Score Silhouette",
            yaxis_title="Échantillons par cluster",
            showlegend=True
        )
        return fig, silhouette_avg



    def create_elbow_plot(self, features, date=None, max_k=8):
        """Plot de la méthode du coude"""
        clustering_data, available_features = self.prepare_data(features, date, normalize=True)
        X = clustering_data[available_features].values
        
        k_range = range(1, min(max_k + 1, len(clustering_data)))
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        fig = px.line(x=list(k_range), y=inertias, 
                     title="Méthode du Coude - Détermination du K optimal",
                     labels={'x': 'Nombre de clusters K', 'y': 'Inertie'})
        
        fig.add_scatter(x=list(k_range), y=inertias, mode='markers')
        
        return fig, inertias


    def create_clustering_visualizations(self, clustering_data, features, method_name):
        """Visualisations complètes intégrant les concepts du cours"""
        X = clustering_data[features].values
        
        # PCA pour visualisation
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        clustering_data['PC1'] = X_pca[:, 0]
        clustering_data['PC2'] = X_pca[:, 1]
        # 1. Scatter plot PCA
        fig_pca = px.scatter(
            clustering_data,
            x='PC1',
            y='PC2',
            color='cluster',
            hover_data=['location'] + features,
            title=f'Clustering {method_name} - Visualisation PCA',
            labels={
                'PC1': f'Composante 1 ({pca.explained_variance_ratio_[0]:.1%})',
                'PC2': f'Composante 2 ({pca.explained_variance_ratio_[1]:.1%})'
            }
        )
        # 2. Carte géographique
        fig_map = px.choropleth(
            clustering_data,
            locations="iso_code",
            color="cluster",
            hover_name="location",
            title=f"Répartition Géographique - {method_name}",
            projection="natural earth"
        )
        
        # 3. Heatmap des centroïdes
        cluster_means = clustering_data.groupby('cluster')[features].mean()
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=cluster_means.values,
            x=cluster_means.index,
            y=cluster_means.columns,
            colorscale='RdYlBu_r'
        ))
        
        fig_heatmap.update_layout(
            title="Centroïdes des Clusters (valeurs standardisées)",
            xaxis_title="Clusters",
            yaxis_title="Features"
        )
        
        return {
            'pca': fig_pca,
            'map': fig_map,
            'heatmap': fig_heatmap
        }