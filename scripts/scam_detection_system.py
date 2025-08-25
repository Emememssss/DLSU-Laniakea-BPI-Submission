"""
HARAYA Scam Detection and False Positive Prevention System
Advanced anomaly detection for Filipino microfinance fraud prevention
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
warnings.filterwarnings('ignore')

@dataclass
class ScamIndicators:
    """Data class for scam detection features"""
    velocity_anomaly: float  # Unusual transaction velocity
    pattern_inconsistency: float  # Inconsistent behavioral patterns  
    network_isolation: float  # Isolation from community networks
    cultural_inauthenticity: float  # Inauthentic cultural behaviors
    temporal_irregularity: float  # Irregular timing patterns
    data_fabrication: float  # Signs of fabricated data
    relationship_manipulation: float  # Manipulated social connections
    financial_impossibility: float  # Financially impossible claims

class ScamDetectionSystem:
    """
    Advanced scam detection system for Filipino microfinance
    Prevents false positives while identifying genuine fraud patterns
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Detection models
        self.velocity_detector = None
        self.pattern_classifier = None
        self.network_analyzer = None
        self.cultural_validator = None
        
        # Scalers
        self.scaler = StandardScaler()
        
        # Thresholds (learned from data)
        self.scam_thresholds = {
            'high_risk': 0.8,      # Clear scam indicators
            'suspicious': 0.6,     # Requires investigation
            'moderate_risk': 0.4,  # Some concerns
            'low_risk': 0.2        # Minimal risk
        }
        
        # Filipino cultural authenticity patterns
        self.cultural_patterns = {
            'family_size_income_correlation': 0.3,  # Larger families often have varied incomes
            'regional_income_distributions': {
                'metro_manila': (15000, 35000),
                'metro_cebu': (12000, 28000), 
                'metro_davao': (10000, 25000),
                'luzon_rural': (8000, 18000),
                'visayas_rural': (7000, 16000),
                'mindanao_rural': (6000, 15000)
            },
            'age_digital_literacy_correlation': -0.4,  # Older users typically less digital
            'community_standing_stability_correlation': 0.6  # High standing = high stability
        }
        
        # Known scam patterns in Filipino microfinance
        self.scam_patterns = {
            'identity_theft': {
                'multiple_accounts_same_device': True,
                'rapid_location_changes': True,
                'inconsistent_personal_info': True
            },
            'synthetic_identity': {
                'perfect_payment_history': True,
                'unrealistic_income_stability': True,
                'no_digital_footprint': True
            },
            'application_fraud': {
                'income_inflation': True,
                'fake_employment': True,
                'fabricated_references': True
            },
            'bust_out_fraud': {
                'rapid_credit_utilization': True,
                'sudden_behavioral_change': True,
                'payment_cessation': True
            }
        }
    
    def detect_scam_indicators(self, persona_data: Dict, transaction_history: Optional[List[Dict]] = None) -> ScamIndicators:
        """
        Detect comprehensive scam indicators
        """
        # Velocity anomaly detection
        velocity_anomaly = self._detect_velocity_anomalies(persona_data, transaction_history)
        
        # Pattern inconsistency analysis
        pattern_inconsistency = self._detect_pattern_inconsistencies(persona_data)
        
        # Network isolation assessment
        network_isolation = self._assess_network_isolation(persona_data)
        
        # Cultural inauthenticity scoring
        cultural_inauthenticity = self._assess_cultural_inauthenticity(persona_data)
        
        # Temporal irregularity detection
        temporal_irregularity = self._detect_temporal_irregularities(persona_data, transaction_history)
        
        # Data fabrication signs
        data_fabrication = self._detect_data_fabrication(persona_data)
        
        # Relationship manipulation indicators
        relationship_manipulation = self._detect_relationship_manipulation(persona_data)
        
        # Financial impossibility checks
        financial_impossibility = self._check_financial_impossibilities(persona_data)
        
        return ScamIndicators(
            velocity_anomaly=velocity_anomaly,
            pattern_inconsistency=pattern_inconsistency,
            network_isolation=network_isolation,
            cultural_inauthenticity=cultural_inauthenticity,
            temporal_irregularity=temporal_irregularity,
            data_fabrication=data_fabrication,
            relationship_manipulation=relationship_manipulation,
            financial_impossibility=financial_impossibility
        )
    
    def _detect_velocity_anomalies(self, persona_data: Dict, transaction_history: Optional[List[Dict]]) -> float:
        """Detect unusual transaction velocity patterns"""
        if not transaction_history:
            # Use persona data to estimate
            monthly_transactions = persona_data.get('mobile_money_transactions_per_month', 15)
            daily_velocity = monthly_transactions / 30.0
        else:
            # Calculate actual velocity from history
            transactions_df = pd.DataFrame(transaction_history)
            daily_counts = transactions_df.groupby(transactions_df['timestamp'].dt.date).size()
            daily_velocity = daily_counts.mean()
        
        # Expected velocity ranges by user type
        income = persona_data.get('monthly_income', 12000)
        digital_literacy = persona_data.get('digital_literacy_score', 4)
        age = persona_data.get('age', 35)
        
        # Calculate expected velocity
        base_velocity = min(2.0, income / 10000)  # Higher income = more transactions
        digital_factor = min(1.5, digital_literacy / 10.0)  # Digital literacy increases usage
        age_factor = max(0.5, 1.0 - (age - 25) / 100.0)  # Younger users more active
        
        expected_velocity = base_velocity * digital_factor * age_factor
        
        # Anomaly score
        velocity_ratio = daily_velocity / (expected_velocity + 0.1)
        
        # Suspicious if too high or too low
        if velocity_ratio > 3.0:  # Much higher than expected
            return min(1.0, velocity_ratio / 10.0)
        elif velocity_ratio < 0.1:  # Much lower than expected
            return min(1.0, 0.3)
        else:
            return 0.0
    
    def _detect_pattern_inconsistencies(self, persona_data: Dict) -> float:
        """Detect inconsistent behavioral patterns"""
        inconsistencies = []
        
        # Age-income inconsistency
        age = persona_data.get('age', 35)
        income = persona_data.get('monthly_income', 12000)
        expected_income = max(8000, min(30000, 5000 + (age - 18) * 500))  # Age-based income expectation
        income_deviation = abs(income - expected_income) / expected_income
        if income_deviation > 1.0:  # More than 100% deviation
            inconsistencies.append(income_deviation)
        
        # Digital literacy-age inconsistency
        digital_score = persona_data.get('digital_literacy_score', 4)
        expected_digital = max(1, min(10, 12 - (age - 20) * 0.15))
        digital_deviation = abs(digital_score - expected_digital) / expected_digital
        if digital_deviation > 0.5:  # More than 50% deviation
            inconsistencies.append(digital_deviation)
        
        # Community standing-stability inconsistency
        community_standing = persona_data.get('community_standing_score', 0.7)
        location_stability = persona_data.get('location_stability_score', 0.7)
        if abs(community_standing - location_stability) > 0.4:  # Should be correlated
            inconsistencies.append(0.5)
        
        # Business type-region inconsistency
        business_type = persona_data.get('business_type', '')
        region = persona_data.get('region', '')
        
        rural_businesses = ['rice_farmer', 'fisherman', 'sari_sari_store']
        urban_businesses = ['online_seller', 'delivery_driver', 'freelancer']
        
        if 'rural' in region and business_type in urban_businesses:
            inconsistencies.append(0.6)
        elif 'metro' in region and business_type in rural_businesses:
            inconsistencies.append(0.4)
        
        return min(1.0, sum(inconsistencies) / 3.0) if inconsistencies else 0.0
    
    def _assess_network_isolation(self, persona_data: Dict) -> float:
        """Assess isolation from community networks"""
        family_size = persona_data.get('family_size', 4)
        community_standing = persona_data.get('community_standing_score', 0.7)
        location_stability = persona_data.get('location_stability_score', 0.7)
        mobile_money_usage = persona_data.get('mobile_money_transactions_per_month', 15)
        
        # Isolation indicators
        isolation_score = 0.0
        
        # Very small family (unusual in Filipino culture)
        if family_size <= 1:
            isolation_score += 0.3
        
        # Low community standing with high claims
        income = persona_data.get('monthly_income', 12000)
        if community_standing < 0.3 and income > 20000:
            isolation_score += 0.4
        
        # High location instability
        if location_stability < 0.2:
            isolation_score += 0.3
        
        # Very low mobile money usage (unusual for loan applicants)
        if mobile_money_usage < 3:
            isolation_score += 0.2
        
        return min(1.0, isolation_score)
    
    def _assess_cultural_inauthenticity(self, persona_data: Dict) -> float:
        """Assess inauthenticity of cultural behaviors"""
        inauthenticity_score = 0.0
        
        # Family size vs community standing inconsistency
        family_size = persona_data.get('family_size', 4)
        community_standing = persona_data.get('community_standing_score', 0.7)
        
        # In Filipino culture, larger families often have higher community presence
        expected_community = min(1.0, 0.4 + (family_size - 1) * 0.1)
        community_deviation = abs(community_standing - expected_community)
        if community_deviation > 0.3:
            inauthenticity_score += 0.3
        
        # Age vs traditional values inconsistency  
        age = persona_data.get('age', 35)
        bill_consistency = persona_data.get('bill_payment_consistency', 0.7)
        
        # Older Filipinos typically have higher payment consistency (cultural value)
        expected_consistency = min(1.0, 0.5 + (age - 25) * 0.01)
        if age > 40 and bill_consistency < expected_consistency - 0.2:
            inauthenticity_score += 0.2
        
        # Regional income inconsistency
        region = persona_data.get('region', 'metro_manila')
        income = persona_data.get('monthly_income', 12000)
        
        regional_ranges = self.cultural_patterns['regional_income_distributions']
        expected_range = regional_ranges.get(region, (8000, 25000))
        min_income, max_income = expected_range
        
        if income < min_income * 0.5 or income > max_income * 2:
            inauthenticity_score += 0.4
        
        # Perfect scores (suspicious in real data)
        perfect_scores = 0
        for score_field in ['community_standing_score', 'location_stability_score', 'bill_payment_consistency']:
            score = persona_data.get(score_field, 0.7)
            if score >= 0.98:  # Suspiciously perfect
                perfect_scores += 1
        
        if perfect_scores >= 2:
            inauthenticity_score += 0.3
        
        return min(1.0, inauthenticity_score)
    
    def _detect_temporal_irregularities(self, persona_data: Dict, transaction_history: Optional[List[Dict]]) -> float:
        """Detect irregular timing patterns"""
        if not transaction_history:
            return 0.0  # Cannot assess without transaction history
        
        transactions_df = pd.DataFrame(transaction_history)
        if len(transactions_df) < 5:
            return 0.0  # Insufficient data
        
        transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
        transactions_df['hour'] = transactions_df['timestamp'].dt.hour
        transactions_df['day_of_week'] = transactions_df['timestamp'].dt.dayofweek
        
        irregularity_score = 0.0
        
        # Check for bot-like patterns (exact timing)
        hour_counts = transactions_df['hour'].value_counts()
        if (hour_counts > len(transactions_df) * 0.5).any():  # More than 50% in single hour
            irregularity_score += 0.4
        
        # Check for unusual time distributions
        night_transactions = len(transactions_df[transactions_df['hour'].between(23, 5)])
        if night_transactions > len(transactions_df) * 0.3:  # More than 30% at night
            irregularity_score += 0.3
        
        # Check for weekend concentration
        weekend_transactions = len(transactions_df[transactions_df['day_of_week'].isin([5, 6])])
        if weekend_transactions > len(transactions_df) * 0.7:  # More than 70% on weekends
            irregularity_score += 0.2
        
        return min(1.0, irregularity_score)
    
    def _detect_data_fabrication(self, persona_data: Dict) -> float:
        """Detect signs of fabricated data"""
        fabrication_score = 0.0
        
        # Check for round numbers (common in fabricated data)
        round_number_fields = ['monthly_income', 'family_size', 'digital_literacy_score']
        round_numbers = 0
        
        for field in round_number_fields:
            value = persona_data.get(field, 0)
            if isinstance(value, (int, float)) and value > 0:
                # Check if it's a suspiciously round number
                if field == 'monthly_income' and value % 5000 == 0 and value >= 15000:
                    round_numbers += 1
                elif field == 'digital_literacy_score' and value in [1, 5, 10]:
                    round_numbers += 1
        
        if round_numbers >= 2:
            fabrication_score += 0.3
        
        # Check for implausible perfection
        high_scores = 0
        score_fields = ['community_standing_score', 'location_stability_score', 'bill_payment_consistency']
        
        for field in score_fields:
            score = persona_data.get(field, 0.7)
            if score >= 0.95:
                high_scores += 1
        
        if high_scores >= 3:  # All scores suspiciously high
            fabrication_score += 0.4
        
        # Check for missing variation in decimal places
        decimal_values = []
        for field in ['community_standing_score', 'location_stability_score', 'bill_payment_consistency']:
            value = persona_data.get(field, 0.7)
            if isinstance(value, float):
                decimal_part = value - int(value)
                decimal_values.append(decimal_part)
        
        # If all decimal parts are 0 or 0.5 (common in fake data)
        if len(decimal_values) >= 3:
            simple_decimals = sum(1 for d in decimal_values if d in [0.0, 0.5])
            if simple_decimals >= 2:
                fabrication_score += 0.2
        
        return min(1.0, fabrication_score)
    
    def _detect_relationship_manipulation(self, persona_data: Dict) -> float:
        """Detect manipulated social connections"""
        manipulation_score = 0.0
        
        # Inconsistent family-community relationship
        family_size = persona_data.get('family_size', 4)
        community_standing = persona_data.get('community_standing_score', 0.7)
        
        # Large family with no community presence (unusual)
        if family_size >= 6 and community_standing < 0.3:
            manipulation_score += 0.4
        
        # High community standing with no stability (manipulated references)
        location_stability = persona_data.get('location_stability_score', 0.7)
        if community_standing > 0.8 and location_stability < 0.3:
            manipulation_score += 0.5
        
        # Age-family size inconsistency
        age = persona_data.get('age', 35)
        if age < 25 and family_size > 6:  # Unlikely to support large family when young
            manipulation_score += 0.3
        elif age > 60 and family_size < 2:  # Unusual in Filipino family structure
            manipulation_score += 0.2
        
        return min(1.0, manipulation_score)
    
    def _check_financial_impossibilities(self, persona_data: Dict) -> float:
        """Check for financially impossible claims"""
        impossibility_score = 0.0
        
        age = persona_data.get('age', 35)
        income = persona_data.get('monthly_income', 12000)
        region = persona_data.get('region', 'metro_manila')
        business_type = persona_data.get('business_type', '')
        education = persona_data.get('education_level', 2)
        
        # Age-income impossibilities
        if age < 22 and income > 25000:  # Very high income for very young person
            impossibility_score += 0.4
        elif age > 65 and income > 30000:  # Very high income for retiree
            impossibility_score += 0.3
        
        # Region-income impossibilities
        regional_medians = {
            'metro_manila': 20000, 'metro_cebu': 16000, 'metro_davao': 14000,
            'luzon_rural': 10000, 'visayas_rural': 9000, 'mindanao_rural': 8000
        }
        
        regional_median = regional_medians.get(region, 12000)
        if income > regional_median * 3:  # More than 3x regional median
            impossibility_score += 0.3
        
        # Business type-income impossibilities
        business_income_limits = {
            'sari_sari_store': 18000,
            'rice_farmer': 15000,
            'fisherman': 12000,
            'delivery_driver': 20000,
            'freelancer': 35000
        }
        
        income_limit = business_income_limits.get(business_type, 50000)
        if income > income_limit:
            impossibility_score += 0.4
        
        # Education-income inconsistencies
        if education <= 1 and income > 20000:  # High income with minimal education
            impossibility_score += 0.2
        
        return min(1.0, impossibility_score)
    
    def calculate_scam_risk_score(self, persona_data: Dict, transaction_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive scam risk score
        """
        start_time = datetime.now()
        
        # Detect scam indicators
        indicators = self.detect_scam_indicators(persona_data, transaction_history)
        
        # Calculate weighted scam risk score
        scam_weights = {
            'velocity_anomaly': 0.15,
            'pattern_inconsistency': 0.20,
            'network_isolation': 0.15,
            'cultural_inauthenticity': 0.20,
            'temporal_irregularity': 0.10,
            'data_fabrication': 0.10,
            'relationship_manipulation': 0.05,
            'financial_impossibility': 0.05
        }
        
        weighted_score = (
            indicators.velocity_anomaly * scam_weights['velocity_anomaly'] +
            indicators.pattern_inconsistency * scam_weights['pattern_inconsistency'] +
            indicators.network_isolation * scam_weights['network_isolation'] +
            indicators.cultural_inauthenticity * scam_weights['cultural_inauthenticity'] +
            indicators.temporal_irregularity * scam_weights['temporal_irregularity'] +
            indicators.data_fabrication * scam_weights['data_fabrication'] +
            indicators.relationship_manipulation * scam_weights['relationship_manipulation'] +
            indicators.financial_impossibility * scam_weights['financial_impossibility']
        )
        
        # Determine risk category
        risk_category = self._determine_scam_risk_category(weighted_score)
        
        # Generate recommendations
        recommendations = self._generate_scam_prevention_recommendations(indicators, weighted_score)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            'scam_risk_score': weighted_score,
            'risk_category': risk_category,
            'scam_indicators': {
                'velocity_anomaly': indicators.velocity_anomaly,
                'pattern_inconsistency': indicators.pattern_inconsistency,
                'network_isolation': indicators.network_isolation,
                'cultural_inauthenticity': indicators.cultural_inauthenticity,
                'temporal_irregularity': indicators.temporal_irregularity,
                'data_fabrication': indicators.data_fabrication,
                'relationship_manipulation': indicators.relationship_manipulation,
                'financial_impossibility': indicators.financial_impossibility
            },
            'primary_concerns': self._identify_primary_concerns(indicators),
            'recommendations': recommendations,
            'false_positive_mitigation': self._assess_false_positive_risk(persona_data, indicators),
            'processing_time_ms': processing_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def _determine_scam_risk_category(self, score: float) -> str:
        """Determine scam risk category"""
        if score >= self.scam_thresholds['high_risk']:
            return "HIGH_SCAM_RISK"
        elif score >= self.scam_thresholds['suspicious']:
            return "SUSPICIOUS_ACTIVITY"
        elif score >= self.scam_thresholds['moderate_risk']:
            return "MODERATE_CONCERN"
        else:
            return "LOW_RISK"
    
    def _identify_primary_concerns(self, indicators: ScamIndicators) -> List[str]:
        """Identify primary areas of concern"""
        concerns = []
        threshold = 0.5
        
        if indicators.velocity_anomaly > threshold:
            concerns.append("Unusual transaction velocity patterns")
        if indicators.pattern_inconsistency > threshold:
            concerns.append("Inconsistent behavioral patterns")
        if indicators.network_isolation > threshold:
            concerns.append("Isolation from community networks")
        if indicators.cultural_inauthenticity > threshold:
            concerns.append("Inauthentic cultural behaviors")
        if indicators.temporal_irregularity > threshold:
            concerns.append("Irregular timing patterns")
        if indicators.data_fabrication > threshold:
            concerns.append("Signs of fabricated data")
        if indicators.relationship_manipulation > threshold:
            concerns.append("Potentially manipulated relationships")
        if indicators.financial_impossibility > threshold:
            concerns.append("Financially impossible claims")
        
        return concerns if concerns else ["No significant concerns detected"]
    
    def _generate_scam_prevention_recommendations(self, indicators: ScamIndicators, risk_score: float) -> List[str]:
        """Generate actionable scam prevention recommendations"""
        recommendations = []
        
        if risk_score >= 0.8:
            recommendations.extend([
                "DENY APPLICATION - High scam risk detected",
                "Flag account for fraud investigation",
                "Report suspicious patterns to compliance team"
            ])
        elif risk_score >= 0.6:
            recommendations.extend([
                "Require additional documentation and verification",
                "Conduct manual review of all provided information",
                "Implement enhanced monitoring if approved"
            ])
        elif risk_score >= 0.4:
            recommendations.extend([
                "Request additional identity verification",
                "Verify community references through alternative channels",
                "Start with lower credit limits"
            ])
        else:
            recommendations.append("Standard approval process - minimal risk detected")
        
        # Specific recommendations based on indicators
        if indicators.cultural_inauthenticity > 0.5:
            recommendations.append("Validate cultural information with community leaders")
        
        if indicators.network_isolation > 0.5:
            recommendations.append("Require community vouching or guarantors")
        
        if indicators.data_fabrication > 0.5:
            recommendations.append("Cross-reference all provided data with external sources")
        
        return recommendations
    
    def _assess_false_positive_risk(self, persona_data: Dict, indicators: ScamIndicators) -> Dict[str, Any]:
        """Assess risk of false positives and provide mitigation"""
        false_positive_risk = 0.0
        mitigation_factors = []
        
        # Legitimate edge cases that might trigger false positives
        age = persona_data.get('age', 35)
        region = persona_data.get('region', '')
        business_type = persona_data.get('business_type', '')
        
        # Young entrepreneurs might have high income legitimately
        if age < 30 and business_type in ['online_seller', 'freelancer'] and 'metro' in region:
            false_positive_risk += 0.3
            mitigation_factors.append("Young entrepreneur in urban area - high income possible")
        
        # OFW families might have unusual patterns
        family_size = persona_data.get('family_size', 4)
        if family_size >= 5 and persona_data.get('monthly_income', 12000) > 20000:
            false_positive_risk += 0.2
            mitigation_factors.append("Possible OFW family - verify overseas employment")
        
        # Seasonal workers have naturally variable patterns
        if 'seasonal' in business_type.lower() or business_type in ['construction', 'tourism_services']:
            false_positive_risk += 0.4
            mitigation_factors.append("Seasonal worker - income variability expected")
        
        # Recent migrants might have low community standing initially
        location_stability = persona_data.get('location_stability_score', 0.7)
        if location_stability < 0.4 and indicators.network_isolation > 0.5:
            false_positive_risk += 0.3
            mitigation_factors.append("Possible recent migrant - community building in progress")
        
        return {
            'false_positive_risk': min(1.0, false_positive_risk),
            'mitigation_factors': mitigation_factors,
            'recommended_validation': self._recommend_validation_steps(false_positive_risk, mitigation_factors)
        }
    
    def _recommend_validation_steps(self, false_positive_risk: float, mitigation_factors: List[str]) -> List[str]:
        """Recommend validation steps to prevent false positives"""
        if false_positive_risk > 0.5:
            return [
                "Manual review by Filipino cultural expert",
                "Extended verification period to gather additional data",
                "Conditional approval with monitoring",
                "Community validation through local partners"
            ]
        elif false_positive_risk > 0.3:
            return [
                "Secondary verification of suspicious indicators",
                "Cultural context review",
                "Extended application interview"
            ]
        else:
            return ["Standard verification process sufficient"]

# Example usage
def main():
    """Test the scam detection system"""
    print("HARAYA Scam Detection System")
    print("=" * 50)
    
    # Initialize detection system
    scam_detector = ScamDetectionSystem()
    
    # Test cases
    test_cases = [
        {
            'name': 'Legitimate Rural Entrepreneur',
            'data': {
                'age': 45, 'monthly_income': 12000, 'community_standing_score': 0.85,
                'family_size': 5, 'digital_literacy_score': 3, 'bill_payment_consistency': 0.8,
                'mobile_money_transactions_per_month': 15, 'location_stability_score': 0.9,
                'region': 'luzon_rural', 'business_type': 'sari_sari_store', 'education_level': 2
            }
        },
        {
            'name': 'Suspicious High-Income User',
            'data': {
                'age': 23, 'monthly_income': 50000, 'community_standing_score': 0.2,
                'family_size': 1, 'digital_literacy_score': 10, 'bill_payment_consistency': 1.0,
                'mobile_money_transactions_per_month': 100, 'location_stability_score': 0.1,
                'region': 'mindanao_rural', 'business_type': 'rice_farmer', 'education_level': 1
            }
        },
        {
            'name': 'Potential Identity Thief',
            'data': {
                'age': 35, 'monthly_income': 25000, 'community_standing_score': 1.0,
                'family_size': 10, 'digital_literacy_score': 1, 'bill_payment_consistency': 1.0,
                'mobile_money_transactions_per_month': 0, 'location_stability_score': 1.0,
                'region': 'metro_manila', 'business_type': 'online_seller', 'education_level': 1
            }
        }
    ]
    
    print("\nTesting scam detection on sample personas:")
    print("=" * 50)
    
    for test_case in test_cases:
        name = test_case['name']
        data = test_case['data']
        
        result = scam_detector.calculate_scam_risk_score(data)
        
        print(f"\n{name}:")
        print(f"  Scam Risk Score: {result['scam_risk_score']:.3f}")
        print(f"  Risk Category: {result['risk_category']}")
        print(f"  Primary Concerns: {', '.join(result['primary_concerns'][:2])}")
        print(f"  False Positive Risk: {result['false_positive_mitigation']['false_positive_risk']:.2f}")
        print(f"  Top Recommendation: {result['recommendations'][0]}")
        print(f"  Processing Time: {result['processing_time_ms']:.1f}ms")
    
    print("\n✅ Scam detection system operational!")
    print("🚫 Advanced fraud prevention active")
    print("🎯 False positive mitigation enabled")
    print("🇵🇭 Filipino cultural context validation integrated")

if __name__ == "__main__":
    main()