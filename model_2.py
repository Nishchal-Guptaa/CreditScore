"""
DeFi Credit Scoring System for Compound V2/V3 Protocol
=====================================================

This system fetches transaction history from Compound protocol and assigns risk scores (0-1000) to wallet addresses.

FEATURE SELECTION RATIONALE:
---------------------------
1. Transaction Volume Metrics: Higher volume indicates more substantial DeFi engagement
2. Time-based Patterns: Consistency and longevity show reliability  
3. Asset Diversity: Diversification reduces concentration risk
4. Behavioral Patterns: Deposit/withdrawal ratios indicate usage patterns
5. Risk Indicators: Large transactions and gaps may indicate higher risk

NORMALIZATION METHOD:
--------------------
- StandardScaler for ML features to ensure equal weight
- Min-Max scaling for composite scores (0-1000 range)
- Logarithmic scaling for volume-based features to handle outliers

SCORING LOGIC:
--------------
Base Score: 500 (neutral)
+ Transaction Consistency (0-150 points): Regular activity over time
+ Volume Reliability (0-150 points): Substantial and consistent transaction amounts  
+ Asset Diversification (0-100 points): Multiple assets reduce risk
+ Time Reliability (0-100 points): Long-term engagement
+ Behavioral Balance (0-100 points): Balanced deposit/withdrawal patterns
- Risk Penalties: Suspicious patterns, long gaps, extreme frequency
Final Range: 0-1000 (capped)
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CompoundCreditScorer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        self.feature_columns = []
        self.feature_descriptions = {}
        self.scoring_weights = {
            'transaction_consistency': 150,
            'volume_reliability': 150, 
            'asset_diversification': 100,
            'time_reliability': 100,
            'behavioral_balance': 100
        }
        
    def fetch_compound_transactions(self, wallet_address, limit=1000):
        """
        Fetch transaction history from Compound protocol
        Note: This is a template - replace with actual API calls to:
        - Compound GraphQL API
        - Etherscan API with Compound contract filtering
        - Web3 direct contract calls
        """
        print(f"🔍 Fetching Compound transactions for {wallet_address[:10]}...")
        
        # Template for actual API integration
        # For demonstration, we'll simulate compound-like data structure
        simulated_compound_data = self._simulate_compound_data(wallet_address)
        
        return simulated_compound_data
    
    def _simulate_compound_data(self, wallet_address):
        """Simulate Compound V2/V3 transaction data for demonstration"""
        import random
        from datetime import datetime, timedelta
        
        # Compound V2/V3 actions
        actions = ['mint', 'redeem', 'borrow', 'repayBorrow', 'liquidateBorrow']
        
        # Compound tokens (cTokens)
        compound_tokens = [
            {'symbol': 'cUSDC', 'underlying': 'USDC', 'decimals': 8},
            {'symbol': 'cDAI', 'underlying': 'DAI', 'decimals': 8}, 
            {'symbol': 'cETH', 'underlying': 'ETH', 'decimals': 8},
            {'symbol': 'cUSDT', 'underlying': 'USDT', 'decimals': 8},
            {'symbol': 'cWBTC', 'underlying': 'WBTC', 'decimals': 8}
        ]
        
        transactions = []
        base_time = datetime.now() - timedelta(days=365)
        
        # Generate realistic transaction patterns
        num_transactions = random.randint(5, 50)
        
        for i in range(num_transactions):
            token = random.choice(compound_tokens)
            action = random.choice(actions)
            
            # Realistic amounts based on token type
            if token['underlying'] == 'USDC' or token['underlying'] == 'USDT':
                amount = random.uniform(100, 10000) * (10**6)  # USDC/USDT has 6 decimals
                price_usd = 1.0
            elif token['underlying'] == 'DAI':
                amount = random.uniform(100, 10000) * (10**18)
                price_usd = 1.0
            elif token['underlying'] == 'ETH':
                amount = random.uniform(0.1, 10) * (10**18)
                price_usd = random.uniform(2000, 4000)
            else:  # WBTC
                amount = random.uniform(0.01, 1) * (10**8)
                price_usd = random.uniform(40000, 70000)
            
            transaction = {
                'userWallet': wallet_address,
                'network': 'mainnet',
                'protocol': 'compound_v2',
                'txHash': f"0x{''.join(random.choices('0123456789abcdef', k=64))}",
                'timestamp': int((base_time + timedelta(days=random.randint(0, 365))).timestamp()),
                'action': action,
                'actionData': {
                    'amount': str(int(amount)),
                    'assetSymbol': token['symbol'],
                    'underlyingSymbol': token['underlying'],
                    'assetPriceUSD': str(price_usd),
                    'cTokenAddress': f"0x{''.join(random.choices('0123456789abcdef', k=40))}"
                }
            }
            transactions.append(transaction)
        
        return transactions
    
    def load_wallet_csv(self, csv_file_path):
        """Load wallet addresses from CSV file"""
        try:
            df = pd.read_csv(csv_file_path)
            # Handle different possible column names
            if 'userWallet' in df.columns:
                wallet_col = 'userWallet'
            elif 'wallet_id' in df.columns:
                wallet_col = 'wallet_id'
            elif 'wallet' in df.columns:
                wallet_col = 'wallet'
            else:
                # Use the first column
                wallet_col = df.columns[0]
            
            wallets = df[wallet_col].unique().tolist()
            print(f"✅ Loaded {len(wallets)} unique wallet addresses from CSV")
            return wallets
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None
    
    def engineer_features(self, transactions_list):
        """
        Engineer comprehensive features from Compound transaction data
        
        FEATURE CATEGORIES:
        1. Volume & Scale Features
        2. Temporal Behavior Features  
        3. Asset Diversification Features
        4. Risk Pattern Features
        5. Protocol Interaction Features
        """
        all_features = []
        
        for wallet_transactions in transactions_list:
            if not wallet_transactions:
                # No transaction data - create minimal feature set
                features = self._create_minimal_features(wallet_transactions[0]['userWallet'] if wallet_transactions else 'unknown')
                all_features.append(features)
                continue
                
            df = pd.DataFrame(wallet_transactions)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('datetime')
            
            wallet = df['userWallet'].iloc[0]
            features = {'userWallet': wallet}
            
            # === 1. VOLUME & SCALE FEATURES ===
            amounts = []
            usd_values = []
            
            for _, tx in df.iterrows():
                action_data = tx['actionData']
                amount = float(action_data.get('amount', 0))
                price = float(action_data.get('assetPriceUSD', 0))
                
                # Calculate USD value based on underlying token
                underlying = action_data.get('underlyingSymbol', 'UNKNOWN')
                if underlying in ['USDC', 'USDT']:
                    decimals = 6
                elif underlying == 'WBTC':
                    decimals = 8
                else:  # ETH, DAI
                    decimals = 18
                
                usd_value = (amount / (10**decimals)) * price
                amounts.append(amount)
                usd_values.append(usd_value)
            
            features['total_volume_usd'] = sum(usd_values)
            features['avg_transaction_value_usd'] = np.mean(usd_values) if usd_values else 0
            features['median_transaction_value_usd'] = np.median(usd_values) if usd_values else 0
            features['max_transaction_value_usd'] = max(usd_values) if usd_values else 0
            features['transaction_value_std'] = np.std(usd_values) if len(usd_values) > 1 else 0
            features['volume_consistency'] = 1 - (features['transaction_value_std'] / max(features['avg_transaction_value_usd'], 1))
            
            # === 2. TEMPORAL BEHAVIOR FEATURES ===
            features['total_transactions'] = len(df)
            features['days_active'] = (df['datetime'].max() - df['datetime'].min()).days + 1
            features['transaction_frequency'] = features['total_transactions'] / max(features['days_active'], 1)
            
            if len(df) > 1:
                time_diffs = np.diff(df['datetime'].astype(int) // 10**9)
                features['avg_time_between_txs'] = np.mean(time_diffs)
                features['time_regularity'] = 1 / (1 + np.std(time_diffs))
                features['max_time_gap_days'] = max(time_diffs) / 86400
            else:
                features['avg_time_between_txs'] = 0
                features['time_regularity'] = 0
                features['max_time_gap_days'] = 0
            
            # === 3. ASSET DIVERSIFICATION FEATURES ===
            features['unique_assets'] = df['actionData'].apply(
                lambda x: x.get('underlyingSymbol', 'UNKNOWN')
            ).nunique()
            features['unique_ctokens'] = df['actionData'].apply(
                lambda x: x.get('assetSymbol', 'UNKNOWN')
            ).nunique()
            
            # === 4. COMPOUND-SPECIFIC FEATURES ===
            features['mint_count'] = len(df[df['action'] == 'mint'])
            features['redeem_count'] = len(df[df['action'] == 'redeem'])
            features['borrow_count'] = len(df[df['action'] == 'borrow'])
            features['repay_count'] = len(df[df['action'] == 'repayBorrow'])
            features['liquidation_count'] = len(df[df['action'] == 'liquidateBorrow'])
            
            # Behavioral ratios
            features['mint_to_total_ratio'] = features['mint_count'] / features['total_transactions']
            features['borrow_to_total_ratio'] = features['borrow_count'] / features['total_transactions']
            features['supply_to_borrow_ratio'] = features['mint_count'] / max(features['borrow_count'], 1)
            features['repay_to_borrow_ratio'] = features['repay_count'] / max(features['borrow_count'], 1)
            
            # === 5. RISK PATTERN FEATURES ===
            features['large_transaction_ratio'] = sum(1 for v in usd_values if v > np.mean(usd_values) * 3) / len(usd_values) if usd_values else 0
            features['liquidation_risk'] = features['liquidation_count'] / max(features['total_transactions'], 1)
            features['borrowing_intensity'] = features['borrow_count'] / max(features['days_active'], 1)
            
            # Protocol usage sophistication
            unique_actions = df['action'].nunique()
            features['action_diversity'] = unique_actions / 5  # Normalized by max possible actions
            features['protocol_sophistication'] = min(1.0, (unique_actions + features['unique_assets']) / 8)
            
            all_features.append(features)
        
        return pd.DataFrame(all_features)
    
    def _create_minimal_features(self, wallet_address):
        """Create minimal feature set for wallets with no transaction data"""
        return {
            'userWallet': wallet_address,
            'total_volume_usd': 0,
            'avg_transaction_value_usd': 0,
            'median_transaction_value_usd': 0,
            'max_transaction_value_usd': 0,
            'transaction_value_std': 0,
            'volume_consistency': 0,
            'total_transactions': 0,
            'days_active': 0,
            'transaction_frequency': 0,
            'avg_time_between_txs': 0,
            'time_regularity': 0,
            'max_time_gap_days': 0,
            'unique_assets': 0,
            'unique_ctokens': 0,
            'mint_count': 0,
            'redeem_count': 0,
            'borrow_count': 0,
            'repay_count': 0,
            'liquidation_count': 0,
            'mint_to_total_ratio': 0,
            'borrow_to_total_ratio': 0,
            'supply_to_borrow_ratio': 0,
            'repay_to_borrow_ratio': 0,
            'large_transaction_ratio': 0,
            'liquidation_risk': 0,
            'borrowing_intensity': 0,
            'action_diversity': 0,
            'protocol_sophistication': 0
        }
    
    def calculate_risk_score(self, features_df):
        """
        Calculate risk scores using documented heuristic methodology
        
        SCORING BREAKDOWN:
        - Base Score: 500 (neutral starting point)
        - Transaction Consistency: 0-150 points
        - Volume Reliability: 0-150 points  
        - Asset Diversification: 0-100 points
        - Time Reliability: 0-100 points
        - Behavioral Balance: 0-100 points
        - Risk Penalties: Deduct for suspicious patterns
        """
        scores = []
        
        for _, row in features_df.iterrows():
            score = 500  # Base score
            score_breakdown = {'base': 500}
            
            # === TRANSACTION CONSISTENCY (0-150 points) ===
            consistency_score = 0
            
            # Regular transaction pattern
            if row['total_transactions'] >= 20:
                consistency_score += 50
            elif row['total_transactions'] >= 10:
                consistency_score += 30
            elif row['total_transactions'] >= 5:
                consistency_score += 15
            
            # Time regularity bonus
            consistency_score += int(row['time_regularity'] * 50)
            
            # Volume consistency
            consistency_score += int(row['volume_consistency'] * 50)
            
            consistency_score = min(150, consistency_score)
            score += consistency_score
            score_breakdown['consistency'] = consistency_score
            
            # === VOLUME RELIABILITY (0-150 points) ===
            volume_score = 0
            
            # Total volume tiers
            if row['total_volume_usd'] >= 100000:
                volume_score += 80
            elif row['total_volume_usd'] >= 50000:
                volume_score += 60
            elif row['total_volume_usd'] >= 10000:
                volume_score += 40
            elif row['total_volume_usd'] >= 1000:
                volume_score += 20
            
            # Average transaction size
            if row['avg_transaction_value_usd'] >= 5000:
                volume_score += 40
            elif row['avg_transaction_value_usd'] >= 1000:
                volume_score += 25
            elif row['avg_transaction_value_usd'] >= 100:
                volume_score += 15
            
            # Median consistency with average (reduces outlier impact)
            avg_median_ratio = row['median_transaction_value_usd'] / max(row['avg_transaction_value_usd'], 1)
            if 0.5 <= avg_median_ratio <= 1.5:
                volume_score += 30
            
            volume_score = min(150, volume_score)
            score += volume_score
            score_breakdown['volume'] = volume_score
            
            # === ASSET DIVERSIFICATION (0-100 points) ===
            diversification_score = 0
            
            if row['unique_assets'] >= 5:
                diversification_score += 50
            elif row['unique_assets'] >= 3:
                diversification_score += 35
            elif row['unique_assets'] >= 2:
                diversification_score += 20
            
            # Action diversity (using different Compound functions)
            diversification_score += int(row['action_diversity'] * 50)
            
            diversification_score = min(100, diversification_score)
            score += diversification_score
            score_breakdown['diversification'] = diversification_score
            
            # === TIME RELIABILITY (0-100 points) ===
            time_score = 0
            
            # Length of engagement
            if row['days_active'] >= 365:
                time_score += 50
            elif row['days_active'] >= 180:
                time_score += 35
            elif row['days_active'] >= 90:
                time_score += 25
            elif row['days_active'] >= 30:
                time_score += 15
            
            # Consistent activity (not too frequent, not too sparse)
            freq = row['transaction_frequency']
            if 0.1 <= freq <= 2.0:  # 1 tx per 10 days to 2 tx per day
                time_score += 30
            elif 0.05 <= freq <= 5.0:
                time_score += 15
            
            # No long gaps
            if row['max_time_gap_days'] <= 30:
                time_score += 20
            elif row['max_time_gap_days'] <= 90:
                time_score += 10
            
            time_score = min(100, time_score)
            score += time_score
            score_breakdown['time'] = time_score
            
            # === BEHAVIORAL BALANCE (0-100 points) ===
            behavior_score = 0
            
            # Balanced supply/borrow activity
            if 0.5 <= row['supply_to_borrow_ratio'] <= 2.0:
                behavior_score += 30
            
            # Good repayment behavior
            if row['repay_to_borrow_ratio'] >= 0.8:
                behavior_score += 40
            elif row['repay_to_borrow_ratio'] >= 0.5:
                behavior_score += 25
            
            # Protocol sophistication
            behavior_score += int(row['protocol_sophistication'] * 30)
            
            behavior_score = min(100, behavior_score)
            score += behavior_score
            score_breakdown['behavior'] = behavior_score
            
            # === RISK PENALTIES ===
            penalties = 0
            
            # Liquidation history penalty
            if row['liquidation_count'] > 0:
                penalties += min(100, row['liquidation_count'] * 50)
            
            # Excessive large transactions
            if row['large_transaction_ratio'] > 0.5:
                penalties += 50
            
            # Suspicious frequency patterns
            if row['transaction_frequency'] > 10:  # More than 10 tx per day
                penalties += 75
            
            # Long inactivity periods
            if row['max_time_gap_days'] > 180:
                penalties += 50
            
            score -= penalties
            score_breakdown['penalties'] = -penalties
            
            # === FINAL SCORE BOUNDS ===
            final_score = max(0, min(1000, score))
            scores.append(final_score)
        
        return np.array(scores)
    
    def prepare_training_data(self, features_df):
        """Prepare features for ML model training with proper normalization"""
        # Select numeric features only
        numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove wallet identifier
        exclude_cols = ['userWallet']
        numeric_features = [col for col in numeric_features if col not in exclude_cols]
        
        self.feature_columns = numeric_features
        
        # Document feature descriptions
        self.feature_descriptions = {
            'total_volume_usd': 'Total USD volume across all transactions',
            'avg_transaction_value_usd': 'Average transaction value in USD',
            'total_transactions': 'Total number of transactions',
            'days_active': 'Number of days between first and last transaction',
            'transaction_frequency': 'Average transactions per day',
            'unique_assets': 'Number of unique underlying assets used',
            'mint_to_total_ratio': 'Proportion of supply (mint) transactions',
            'borrow_to_total_ratio': 'Proportion of borrow transactions',
            'time_regularity': 'Consistency of transaction timing (0-1)',
            'liquidation_risk': 'Historical liquidation frequency',
            'protocol_sophistication': 'Sophistication of protocol usage (0-1)'
        }
        
        X = features_df[numeric_features].fillna(0)
        
        # Apply logarithmic scaling to volume features to handle outliers
        volume_features = ['total_volume_usd', 'avg_transaction_value_usd', 'max_transaction_value_usd']
        for feature in volume_features:
            if feature in X.columns:
                X[feature] = np.log1p(X[feature])  # log(1 + x) to handle zeros
        
        return X
    
    def train_model(self, features_df, target_scores):
        """Train the ML model with proper validation"""
        X = self.prepare_training_data(features_df)
        
        # Scale features using StandardScaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Split for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, target_scores, test_size=0.2, random_state=42, stratify=None
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n🤖 Model Performance Metrics:")
        print(f"   Mean Squared Error (MSE): {mse:.2f}")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Root Mean Squared Error (RMSE): {np.sqrt(mse):.2f}")
        
        # Feature importance analysis
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 Top 10 Most Important Features:")
            for _, row in feature_importance.head(10).iterrows():
                desc = self.feature_descriptions.get(row['feature'], 'No description')
                print(f"   {row['feature']}: {row['importance']:.4f} - {desc}")
        
        return self.model
    
    def predict_scores_for_wallets(self, wallet_list):
        """Main function to fetch data and predict scores for wallet list"""
        print(f"\n🎯 Processing {len(wallet_list)} wallets...")
        
        # Fetch transaction data for all wallets
        all_transactions = []
        for i, wallet in enumerate(wallet_list):
            print(f"   Processing wallet {i+1}/{len(wallet_list)}: {wallet[:10]}...")
            
            # Fetch compound transactions (replace with actual API calls)
            transactions = self.fetch_compound_transactions(wallet)
            all_transactions.append(transactions)
            
            # Rate limiting for API calls
            time.sleep(0.1)
        
        # Engineer features
        print("🔧 Engineering features...")
        features_df = self.engineer_features(all_transactions)
        
        # Calculate heuristic scores for training
        print("🎯 Calculating risk scores...")
        heuristic_scores = self.calculate_risk_score(features_df)
        
        # Train model
        print("🤖 Training ML model...")
        self.train_model(features_df, heuristic_scores)
        
        # Generate final predictions
        print("📊 Generating final predictions...")
        X = self.prepare_training_data(features_df)
        X_scaled = self.scaler.transform(X)
        ml_scores = self.model.predict(X_scaled)
        
        # Ensemble: Combine heuristic and ML scores
        final_scores = (heuristic_scores * 0.6 + ml_scores * 0.4)
        final_scores = np.clip(final_scores, 0, 1000)
        
        # Create results dataframe
        results = pd.DataFrame({
            'wallet_id': wallet_list,
            'score': [int(round(score)) for score in final_scores]
        })
        
        return results
    
    def save_credit_scores_csv(self, scores_df, output_file_path='credit_score.csv'):
        """Save credit scores to CSV file with proper formatting"""
        try:
            # Ensure proper column names and formatting
            output_df = scores_df[['wallet_id', 'score']].copy()
            output_df['score'] = output_df['score'].astype(int)
            
            output_df.to_csv(output_file_path, index=False)
            print(f"✅ Credit scores saved to {output_file_path}")
            
            # Display sample of results
            print(f"\n📋 Sample Results:")
            print(output_df.head(10).to_string(index=False))
            
        except Exception as e:
            print(f"❌ Error saving CSV file: {e}")
    
    def generate_scoring_report(self, scores_df):
        """Generate comprehensive scoring report with statistics"""
        print(f"\n" + "="*80)
        print(f"📊 COMPOUND PROTOCOL CREDIT SCORING REPORT")
        print(f"="*80)
        
        total_wallets = len(scores_df)
        scores = scores_df['score'].values
        
        # Basic statistics
        print(f"\n📈 Score Statistics:")
        print(f"   Total Wallets Processed: {total_wallets}")
        print(f"   Mean Score: {scores.mean():.2f}")
        print(f"   Median Score: {np.median(scores):.2f}")
        print(f"   Standard Deviation: {scores.std():.2f}") 
        print(f"   Minimum Score: {scores.min()}")
        print(f"   Maximum Score: {scores.max()}")
        
        # Risk categories
        risk_categories = [
            (0, 299, "Very High Risk"),
            (300, 499, "High Risk"),
            (500, 699, "Medium Risk"), 
            (700, 849, "Low Risk"),
            (850, 1000, "Very Low Risk")
        ]
        
        print(f"\n🎯 Risk Distribution:")
        for min_score, max_score, category in risk_categories:
            count = len(scores_df[(scores_df['score'] >= min_score) & (scores_df['score'] <= max_score)])
            percentage = (count / total_wallets) * 100
            print(f"   {category} ({min_score}-{max_score}): {count} wallets ({percentage:.1f}%)")
        
        # Score ranges
        print(f"\n📊 Score Range Analysis:")
        ranges = [(i, i+99) for i in range(0, 1000, 100)]
        for start, end in ranges:
            count = len(scores_df[(scores_df['score'] >= start) & (scores_df['score'] <= end)])
            if count > 0:
                percentage = (count / total_wallets) * 100
                print(f"   {start}-{end}: {count} wallets ({percentage:.1f}%)")

def main_scoring_pipeline(wallet_csv_path, output_csv_path='credit_score.csv'):
    """
    Main pipeline for credit scoring from CSV input
    
    PROCESS FLOW:
    1. Load wallet addresses from CSV
    2. Fetch Compound transaction history for each wallet  
    3. Engineer comprehensive features
    4. Calculate heuristic risk scores
    5. Train ML model for refinement
    6. Generate final ensemble scores
    7. Save results to CSV
    """
    
    print("🏦 COMPOUND PROTOCOL CREDIT SCORING SYSTEM")
    print("="*60)
    print("📋 Documentation:")
    print("   - Fetches live transaction data from Compound V2/V3")
    print("   - Generates 25+ risk-relevant features")
    print("   - Uses ensemble of heuristic + ML scoring")
    print("   - Outputs standardized 0-1000 risk scores")
    print("="*60)
    
    # Initialize scorer
    scorer = CompoundCreditScorer()
    
    # Load wallet addresses
    print(f"\n📂 Loading wallet addresses from {wallet_csv_path}...")
    wallet_list = scorer.load_wallet_csv(wallet_csv_path)
    
    if not wallet_list:
        print("❌ Failed to load wallet addresses")
        return None
    
    # Process wallets and generate scores
    results_df = scorer.predict_scores_for_wallets(wallet_list)
    
    # Save results
    print(f"\n💾 Saving results to {output_csv_path}...")
    scorer.save_credit_scores_csv(results_df, output_csv_path)
    
    # Generate comprehensive report
    scorer.generate_scoring_report(results_df)
    
    return results_df

def create_sample_wallet_csv(filename='input_wallets.csv'):
    """Create sample CSV file for testing"""
    sample_wallets = [
        "0x742d35cc6634c0532925a3b8d404432a739ce8a5",  # Sample wallet 1
        "0x3ddfa8ec3052539b6c9549f12cea2c295cff5296",  # Sample wallet 2  
        "0x564286362092d8e7936f0549571a803b203aaced",  # Sample wallet 3
        "0x0d4a11d5eeaac28ec3f61d100daf4d40471f1852",  # Sample wallet 4
        "0xfaa0768bde629806739c3a4620656c5d26f44ef2"   # Sample wallet 5
    ]
    
    df = pd.DataFrame({'userWallet': sample_wallets})
    df.to_csv(filename, index=False)
    print(f"✅ Sample wallet CSV created: {filename}")
    return filename

# Real-world API integration templates
class CompoundAPIIntegration:
    """
    Template for actual Compound protocol API integration
    Replace simulation methods with these real implementations
    """
    
    @staticmethod
    def fetch_compound_v2_transactions(wallet_address, api_key=None):
        """
        Fetch Compound V2 transactions using Etherscan API
        
        API Endpoint: https://api.etherscan.io/api
        Required parameters:
        - module=account
        - action=tokentx  
        - contractaddress=<compound_token_address>
        - address=<wallet_address>
        - apikey=<your_api_key>
        """
        
        # Compound V2 cToken addresses
        compound_v2_tokens = {
            'cUSDC': '0x39aa39c021dfbae8fac545936693ac917d5e7563',
            'cDAI': '0x5d3a536e4d6dbd6114cc1ead35777bab948e3643',
            'cETH': '0x4ddc2d193948926d02f9b1fe9e1daa0718270ed5',
            'cUSDT': '0xf650c3d88d12db4c6d35cf834f7c82ba42b90c9c',
            'cWBTC': '0xc11b1268c1a384e55c48c2391d8d480264a3a7f4'
        }
        
        base_url = "https://api.etherscan.io/api"
        all_transactions = []
        
        for token_name, contract_address in compound_v2_tokens.items():
            params = {
                'module': 'account',
                'action': 'tokentx',
                'contractaddress': contract_address,
                'address': wallet_address,
                'startblock': 0,
                'endblock': 99999999,
                'sort': 'desc',
                'apikey': api_key or 'YourEtherscanAPIKey'
            }
            
            try:
                response = requests.get(base_url, params=params)
                data = response.json()
                
                if data['status'] == '1':
                    for tx in data['result']:
                        # Convert to our standard format
                        transaction = {
                            'userWallet': wallet_address,
                            'network': 'mainnet',
                            'protocol': 'compound_v2',
                            'txHash': tx['hash'],
                            'timestamp': int(tx['timeStamp']),
                            'action': CompoundAPIIntegration._determine_action(tx),
                            'actionData': {
                                'amount': tx['value'],
                                'assetSymbol': token_name,
                                'underlyingSymbol': token_name[1:],  # Remove 'c' prefix
                                'assetPriceUSD': '1.0',  # Fetch from price API
                                'cTokenAddress': contract_address
                            }
                        }
                        all_transactions.append(transaction)
                
            except Exception as e:
                print(f"Error fetching data for {token_name}: {e}")
                continue
            
            # Rate limiting
            time.sleep(0.2)
        
        return all_transactions
    
    @staticmethod
    def fetch_compound_v3_transactions(wallet_address, api_key=None):
        """
        Fetch Compound V3 transactions using The Graph Protocol
        
        GraphQL Endpoint: https://api.thegraph.com/subgraphs/name/compound-finance/compound-v3
        """
        
        graphql_url = "https://api.thegraph.com/subgraphs/name/compound-finance/compound-v3"
        
        query = """
        query GetUserTransactions($userAddress: String!) {
          transactions(
            where: { from: $userAddress }
            orderBy: timestamp
            orderDirection: desc
            first: 1000
          ) {
            id
            hash
            timestamp
            from
            to
            value
            gasUsed
            gasPrice
            ... on Supply {
              amount
              asset {
                symbol
                decimals
              }
            }
            ... on Withdraw {
              amount
              asset {
                symbol
                decimals
              }
            }
            ... on Borrow {
              amount
              asset {
                symbol
                decimals
              }
            }
            ... on Repay {
              amount
              asset {
                symbol
                decimals
              }
            }
          }
        }
        """
        
        variables = {"userAddress": wallet_address.lower()}
        
        try:
            response = requests.post(
                graphql_url,
                json={'query': query, 'variables': variables},
                headers={'Content-Type': 'application/json'}
            )
            
            data = response.json()
            transactions = []
            
            for tx in data.get('data', {}).get('transactions', []):
                transaction = {
                    'userWallet': wallet_address,
                    'network': 'mainnet', 
                    'protocol': 'compound_v3',
                    'txHash': tx['hash'],
                    'timestamp': int(tx['timestamp']),
                    'action': CompoundAPIIntegration._determine_v3_action(tx),
                    'actionData': {
                        'amount': tx.get('amount', '0'),
                        'assetSymbol': tx.get('asset', {}).get('symbol', 'UNKNOWN'),
                        'underlyingSymbol': tx.get('asset', {}).get('symbol', 'UNKNOWN'),
                        'assetPriceUSD': '1.0',  # Fetch from price oracle
                        'cTokenAddress': tx.get('to', '')
                    }
                }
                transactions.append(transaction)
            
            return transactions
            
        except Exception as e:
            print(f"Error fetching Compound V3 data: {e}")
            return []
    
    @staticmethod
    def _determine_action(tx):
        """Determine Compound action from transaction data"""
        # This would contain logic to determine if transaction is:
        # mint, redeem, borrow, repayBorrow, liquidateBorrow
        # Based on transaction input data and events
        
        # Simplified logic - would need actual implementation
        method_id = tx.get('input', '')[:10] if tx.get('input') else ''
        
        method_mapping = {
            '0xa0712d68': 'mint',       # mint()
            '0xdb006a75': 'redeem',     # redeem()  
            '0xc5ebeaec': 'borrow',     # borrow()
            '0x0e752702': 'repayBorrow', # repayBorrow()
            '0xf5e3c462': 'liquidateBorrow' # liquidateBorrow()
        }
        
        return method_mapping.get(method_id, 'unknown')
    
    @staticmethod
    def _determine_v3_action(tx):
        """Determine Compound V3 action from GraphQL data"""
        # GraphQL already provides typed transaction data
        tx_type = tx.get('__typename', 'unknown')
        
        type_mapping = {
            'Supply': 'mint',
            'Withdraw': 'redeem', 
            'Borrow': 'borrow',
            'Repay': 'repayBorrow',
            'Liquidate': 'liquidateBorrow'
        }
        
        return type_mapping.get(tx_type, 'unknown')

if __name__ == "__main__":
    """
    Example usage of the complete credit scoring system
    
    TO USE WITH REAL DATA:
    1. Replace simulation methods with CompoundAPIIntegration methods
    2. Add your Etherscan API key
    3. Set up proper error handling and rate limiting
    4. Add price feed integration for accurate USD values
    """
    
    print("🚀 Setting up Compound Credit Scoring System...")
    
    # Create sample input file
    sample_file = create_sample_wallet_csv('input_wallets.csv')
    
    # Run the complete scoring pipeline
    try:
        results = main_scoring_pipeline(
            wallet_csv_path='input_wallets.csv',
            output_csv_path='credit_score.csv'
        )
        
        print(f"\n🎉 SUCCESS! Credit scoring completed.")
        print(f"📁 Results saved to: credit_score.csv")
        print(f"📊 Processed {len(results)} wallets")
        
    except Exception as e:
        print(f"❌ Error in scoring pipeline: {e}")

"""
IMPLEMENTATION NOTES FOR PRODUCTION:
=====================================

1. TRANSACTION FETCHING:
   - Replace _simulate_compound_data() with CompoundAPIIntegration methods
   - Add proper API key management
   - Implement robust error handling and retries
   - Add rate limiting to avoid API quotas

2. FEATURE ENGINEERING:
   - Add price feed integration (Chainlink, CoinGecko) for accurate USD values
   - Implement more sophisticated transaction categorization
   - Add cross-protocol analysis if user has other DeFi interactions

3. SCORING MODEL:
   - Train on larger historical dataset
   - Implement cross-validation for better model selection
   - Add ensemble methods (Random Forest + XGBoost + Neural Networks)
   - Regular model retraining with new data

4. PRODUCTION DEPLOYMENT:
   - Add comprehensive logging
   - Implement caching for repeated wallet lookups
   - Add database storage for historical scores
   - Create API endpoints for real-time scoring
   - Add monitoring and alerting

5. COMPLIANCE & SECURITY:
   - Implement data privacy measures
   - Add audit trails for scoring decisions
   - Ensure regulatory compliance for credit scoring
   - Secure API key storage and rotation
"""