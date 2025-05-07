# blockchain_api.py
import requests
import pandas as pd
import json
import time
from datetime import datetime

class BlockchainDataAPI:
    def __init__(self):
        self.base_url = "https://blockchain.info"
        self.last_processed_block = None
        self.last_update_time = 0
        
    def get_latest_transactions(self, limit=10):
        """
        Fetch latest transactions from blockchain
        
        This combines unconfirmed transactions and latest block transactions
        """
        current_time = time.time()
        transactions = []
        
        # Rate limiting - don't call API more than once every 10 seconds
        if current_time - self.last_update_time < 10:
            return []
            
        self.last_update_time = current_time
        
        # Step 1: Try to get unconfirmed transactions first (most recent)
        try:
            unconfirmed_url = f"{self.base_url}/unconfirmed-transactions?format=json"
            response = requests.get(unconfirmed_url)
            
            if response.status_code == 200:
                data = response.json()
                unconfirmed_txs = data.get('txs', [])
                
                # Process the most recent unconfirmed transactions
                for tx in unconfirmed_txs[:min(limit, len(unconfirmed_txs))]:
                    transaction = self._process_transaction(tx)
                    if transaction:
                        transactions.append(transaction)
        except Exception as e:
            print(f"Error fetching unconfirmed transactions: {e}")
        
        # Step 2: If we need more transactions, get from latest block
        if len(transactions) < limit:
            try:
                # Get latest block hash
                latest_block_url = f"{self.base_url}/latestblock"
                response = requests.get(latest_block_url)
                
                if response.status_code == 200:
                    latest_block = response.json()
                    block_hash = latest_block.get('hash')
                    
                    # Skip if we've already processed this block
                    if block_hash != self.last_processed_block:
                        self.last_processed_block = block_hash
                        
                        # Get full block data
                        block_url = f"{self.base_url}/rawblock/{block_hash}"
                        block_response = requests.get(block_url)
                        
                        if block_response.status_code == 200:
                            block_data = block_response.json()
                            block_txs = block_data.get('tx', [])
                            
                            # Process transactions from this block
                            needed = limit - len(transactions)
                            for tx in block_txs[:min(needed, len(block_txs))]:
                                transaction = self._process_transaction(tx)
                                if transaction:
                                    transactions.append(transaction)
            except Exception as e:
                print(f"Error fetching latest block: {e}")
        
        return transactions
    
    def _process_transaction(self, tx):
        """Process a transaction into the format needed for the dashboard"""
        try:
            # Extract basic transaction info
            tx_hash = tx.get('hash', 'unknown')
            tx_time = tx.get('time', int(time.time()))
            
            # Extract inputs (senders)
            inputs = []
            total_input = 0
            for inp in tx.get('inputs', []):
                if 'prev_out' in inp and 'addr' in inp['prev_out']:
                    inputs.append(inp['prev_out']['addr'])
                    total_input += inp['prev_out'].get('value', 0)
            
            # Extract outputs (receivers)
            outputs = []
            for out in tx.get('out', []):
                if 'addr' in out:
                    outputs.append(out['addr'])
            
            # Skip transactions with no inputs or outputs
            if not inputs or not outputs:
                return None
                
            # Calculate amount in BTC
            amount = total_input / 100000000.0  # Convert satoshis to BTC
            
            # Create transaction object
            transaction = {
                'Transaction_ID': tx_hash,
                'Sender_ID': inputs[0] if inputs else "Unknown",
                'Receiver_ID': outputs[0] if outputs else "Unknown", 
                'Transaction_Amount': amount,
                'Timestamp': datetime.fromtimestamp(tx_time).isoformat(),
                'Time_Display': datetime.fromtimestamp(tx_time).strftime("%H:%M:%S")
            }
            
            return transaction
        
        except Exception as e:
            print(f"Error processing transaction {tx.get('hash', 'unknown')}: {e}")
            return None
    
    def get_transaction_details(self, tx_hash):
        """Get detailed information about a specific transaction"""
        try:
            url = f"{self.base_url}/rawtx/{tx_hash}"
            response = requests.get(url)
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error getting transaction details: {e}")
            return None