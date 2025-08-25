"""
HARAYA ML Processing API
Serverless function for heavy ML computations on Vercel
"""

import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, confusion_matrix, roc_auc_score
import pandas as pd
from http.server import BaseHTTPRequestHandler
import urllib.parse

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        """Handle ML processing requests"""
        try:
            # Set CORS headers
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            
            # Parse request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Route to appropriate handler
            operation = request_data.get('operation')
            
            if operation == 'kmeans':
                result = self.process_kmeans(request_data)
            elif operation == 'neural_network_eval':
                result = self.process_neural_network_eval(request_data)
            elif operation == 'cultural_scoring':
                result = self.process_cultural_scoring(request_data)
            elif operation == 'data_generation':
                result = self.process_data_generation(request_data)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            # Send successful response
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
            
        except Exception as e:
            # Send error response
            self.send_response(500)
            self.end_headers()
            error_response = {
                'error': str(e),
                'message': 'ML processing failed'
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

    def process_kmeans(self, request_data):
        """Process K-Means clustering request"""
        data = np.array(request_data['data'])
        k = request_data.get('k', 3)
        normalize = request_data.get('normalize', True)
        
        if normalize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
        else:
            data_scaled = data
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data_scaled)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(data_scaled, cluster_labels)
        inertia = kmeans.inertia_
        
        # Calculate cluster centers in original space
        if normalize:
            cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        else:
            cluster_centers = kmeans.cluster_centers_
        
        return {
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': cluster_centers.tolist(),
            'silhouette_score': float(silhouette_avg),
            'inertia': float(inertia),
            'n_clusters': k,
            'n_samples': len(data)
        }

    def process_neural_network_eval(self, request_data):
        """Process neural network evaluation metrics"""
        y_true = np.array(request_data['y_true'])
        y_pred = np.array(request_data['y_pred'])
        y_prob = np.array(request_data.get('y_prob', y_pred))
        threshold = request_data.get('threshold', 0.5)
        
        # Convert to binary predictions
        y_pred_binary = (y_prob >= threshold).astype(int)
        y_true_binary = (y_true >= threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate AUC if probabilities are available
        try:
            auc = roc_auc_score(y_true_binary, y_prob)
        except:
            auc = None
        
        # Matthews Correlation Coefficient
        mcc_num = (tp * tn - fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0
        
        return {
            'confusion_matrix': {
                'true_positive': int(tp),
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn)
            },
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'specificity': float(specificity),
                'f1_score': float(f1_score),
                'mcc': float(mcc),
                'auc': float(auc) if auc is not None else None
            },
            'threshold': threshold,
            'sample_size': len(y_true)
        }

    def process_cultural_scoring(self, request_data):
        """Process cultural authenticity scoring"""
        personas = request_data['personas']
        
        results = []
        for persona in personas:
            # Cultural Intelligence Score calculation
            kapwa_score = persona.get('kapwa_score', 0.7)
            bayanihan_score = persona.get('bayanihan_participation', 0.8)
            utang_score = persona.get('utang_na_loob_integrity', 0.75)
            community_score = persona.get('community_standing_score', 0.6)
            
            # Weighted cultural authenticity calculation
            cultural_score = (
                kapwa_score * 0.3 + 
                bayanihan_score * 0.25 + 
                utang_score * 0.25 + 
                community_score * 0.2
            )
            
            # Risk assessment based on cultural patterns
            risk_factors = []
            if kapwa_score < 0.3:
                risk_factors.append('Low community connection')
            if bayanihan_score < 0.3:
                risk_factors.append('Minimal mutual aid participation')
            if utang_score < 0.4:
                risk_factors.append('Weak reciprocity patterns')
            
            # Trust classification
            if cultural_score >= 0.7:
                trust_level = 'highly_trustworthy'
            elif cultural_score >= 0.5:
                trust_level = 'moderately_trustworthy'
            else:
                trust_level = 'low_trustworthiness'
            
            results.append({
                'persona_id': persona.get('persona_id', 'unknown'),
                'cultural_authenticity_score': float(cultural_score),
                'trust_level': trust_level,
                'risk_factors': risk_factors,
                'component_scores': {
                    'kapwa': float(kapwa_score),
                    'bayanihan': float(bayanihan_score),
                    'utang_na_loob': float(utang_score),
                    'community_standing': float(community_score)
                }
            })
        
        return {
            'results': results,
            'summary': {
                'total_personas': len(personas),
                'highly_trustworthy': sum(1 for r in results if r['trust_level'] == 'highly_trustworthy'),
                'moderately_trustworthy': sum(1 for r in results if r['trust_level'] == 'moderately_trustworthy'),
                'low_trustworthiness': sum(1 for r in results if r['trust_level'] == 'low_trustworthiness'),
                'average_cultural_score': float(np.mean([r['cultural_authenticity_score'] for r in results]))
            }
        }

    def process_data_generation(self, request_data):
        """Generate synthetic cultural data"""
        n_samples = request_data.get('n_samples', 100)
        categories = request_data.get('categories', ['legitimate', 'edge_case', 'scammer'])
        
        synthetic_data = []
        
        for i in range(n_samples):
            category = np.random.choice(categories)
            
            if category == 'legitimate':
                # Generate legitimate Filipino personas
                persona = {
                    'persona_id': f'synth_legit_{i}',
                    'name': f'Filipino Person {i}',
                    'kapwa_score': np.random.normal(0.75, 0.15),
                    'bayanihan_participation': np.random.normal(0.8, 0.12),
                    'utang_na_loob_integrity': np.random.normal(0.85, 0.1),
                    'community_standing_score': np.random.normal(0.7, 0.15),
                    'monthly_income': np.random.lognormal(10, 0.5),
                    'family_size': np.random.poisson(4) + 1,
                    'digital_literacy_score': np.random.normal(6, 2),
                    'trustworthiness_label': 'trustworthy',
                    'category': 'legitimate'
                }
            elif category == 'edge_case':
                # Generate edge cases
                persona = {
                    'persona_id': f'synth_edge_{i}',
                    'name': f'Edge Case Person {i}',
                    'kapwa_score': np.random.uniform(0.3, 0.7),
                    'bayanihan_participation': np.random.uniform(0.2, 0.6),
                    'utang_na_loob_integrity': np.random.uniform(0.4, 0.8),
                    'community_standing_score': np.random.uniform(0.3, 0.7),
                    'monthly_income': np.random.lognormal(9, 0.8),
                    'family_size': np.random.poisson(3) + 1,
                    'digital_literacy_score': np.random.normal(5, 2.5),
                    'trustworthiness_label': 'challenging_legitimate',
                    'category': 'edge_case'
                }
            else:  # scammer
                # Generate scammer patterns
                persona = {
                    'persona_id': f'synth_scam_{i}',
                    'name': f'Suspicious Person {i}',
                    'kapwa_score': np.random.uniform(0.1, 0.4),
                    'bayanihan_participation': np.random.uniform(0.0, 0.3),
                    'utang_na_loob_integrity': np.random.uniform(0.1, 0.4),
                    'community_standing_score': np.random.uniform(0.1, 0.4),
                    'monthly_income': np.random.lognormal(8, 1.2),
                    'family_size': np.random.poisson(2) + 1,
                    'digital_literacy_score': np.random.normal(7, 1.5),
                    'trustworthiness_label': 'untrustworthy',
                    'category': 'scammer'
                }
            
            # Clamp values to reasonable ranges
            persona['kapwa_score'] = np.clip(persona['kapwa_score'], 0, 1)
            persona['bayanihan_participation'] = np.clip(persona['bayanihan_participation'], 0, 1)
            persona['utang_na_loob_integrity'] = np.clip(persona['utang_na_loob_integrity'], 0, 1)
            persona['community_standing_score'] = np.clip(persona['community_standing_score'], 0, 1)
            persona['monthly_income'] = max(10000, persona['monthly_income'])
            persona['family_size'] = max(1, int(persona['family_size']))
            persona['digital_literacy_score'] = np.clip(persona['digital_literacy_score'], 1, 10)
            
            synthetic_data.append(persona)
        
        return {
            'synthetic_data': synthetic_data,
            'metadata': {
                'total_samples': n_samples,
                'categories_distribution': {cat: sum(1 for p in synthetic_data if p['category'] == cat) for cat in categories},
                'generation_timestamp': pd.Timestamp.now().isoformat(),
                'features': [
                    'kapwa_score', 'bayanihan_participation', 'utang_na_loob_integrity',
                    'community_standing_score', 'monthly_income', 'family_size', 'digital_literacy_score'
                ]
            }
        }

    def do_GET(self):
        """Handle GET requests - health check and info"""
        if self.path == '/api/ml-processing':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                'status': 'active',
                'service': 'HARAYA ML Processing API',
                'version': '1.0.0',
                'available_operations': [
                    'kmeans', 'neural_network_eval', 'cultural_scoring', 'data_generation'
                ],
                'description': 'Serverless ML processing for HARAYA Cultural Intelligence Platform'
            }
            
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')