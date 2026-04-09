import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime
import json
import sys
import os

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class PredictiveMaintenanceSimulator:
    def __init__(self, api_url="http://localhost:8000", data_path="../data/"):
        self.api_url = api_url
        self.data_path = "data/"
        self.sensor_data = None
        self.current_index = 0
        
    def load_data(self):
        """Load and prepare sensor data from CSV file"""
        try:
            # Find CSV file
            csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
            if not csv_files:
                print(f"{Colors.RED}Hata: {self.data_path} klasöründe CSV dosyası bulunamadı!{Colors.END}")
                return False
            
            # Load data
            df = pd.read_csv(os.path.join(self.data_path, csv_files[0]))
            print(f"{Colors.GREEN}Veri seti yüklendi: {csv_files[0]}{Colors.END}")
            print(f"{Colors.BLUE}Toplam veri sayısı: {len(df)}{Colors.END}")
            
            # Select 100 random rows
            self.sensor_data = df.sample(n=min(100, len(df)), random_state=42).reset_index(drop=True)
            print(f"{Colors.YELLOW}Simülasyon için {len(self.sensor_data)} satır rastgele seçildi{Colors.END}")
            
            return True
            
        except Exception as e:
            print(f"{Colors.RED}Veri yükleme hatası: {e}{Colors.END}")
            return False
    
    def prepare_sensor_payload(self, row):
        """Prepare sensor data for API request"""
        return {
            "Machine_ID": int(row["UDI"]),
            "Air_temperature": float(row["Air temperature [K]"]),
            "Process_temperature": float(row["Process temperature [K]"]),
            "Rotational_speed": float(row["Rotational speed [rpm]"]),
            "Torque": float(row["Torque [Nm]"]),
            "Tool_wear": float(row["Tool wear [min]"]),
            "TWF": int(row["TWF"]),
            "HDF": int(row["HDF"]),
            "PWF": int(row["PWF"]),
            "OSF": int(row["OSF"]),
            "RNF": int(row["RNF"])
        }
    
    def send_prediction_request(self, sensor_payload):
        """Send prediction request to API"""
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=sensor_payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"{Colors.RED}API Hatası: {response.status_code} - {response.text}{Colors.END}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"{Colors.RED}İstek hatası: {e}{Colors.END}")
            return None
    
    def print_status(self, timestamp, payload, result):
        """Print simulation status with color coding"""
        failure_prob = result.get("failure_percentage", 0)
        prediction = result.get("prediction", "Unknown")
        machine_id = payload.get("Machine_ID", "Unknown")
        
        print(f"\n{Colors.CYAN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}Makine ID: {machine_id}{Colors.END}")
        print(f"{Colors.BLUE}Zaman: {timestamp}{Colors.END}")
        print(f"{Colors.MAGENTA}Sensör Verileri:{Colors.END}")
        print(f"  Hava Sıcaklığı: {payload['Air_temperature']:.1f}K")
        print(f"  Proses Sıcaklığı: {payload['Process_temperature']:.1f}K")
        print(f"  Dönüş Hızı: {payload['Rotational_speed']:.0f} rpm")
        print(f"  Tork: {payload['Torque']:.1f} Nm")
        print(f"  Takım Aşınması: {payload['Tool_wear']:.0f} min")
        
        # Color code based on failure probability
        if failure_prob > 80:
            print(f"\n{Colors.RED}{Colors.BOLD}KRİTİK UYARI: Makine {machine_id} arıza vermek üzere!{Colors.END}")
            print(f"{Colors.RED}{Colors.BOLD}Arıza İhtimali: {failure_prob:.2f}%{Colors.END}")
            print(f"{Colors.RED}{Colors.BOLD}Tahmin: {prediction}{Colors.END}")
        elif failure_prob > 50:
            print(f"\n{Colors.YELLOW}UYARI: Yüksek Arıza Riski{Colors.END}")
            print(f"{Colors.YELLOW}Arıza İhtimali: {failure_prob:.2f}%{Colors.END}")
            print(f"{Colors.YELLOW}Tahmin: {prediction}{Colors.END}")
        elif failure_prob > 20:
            print(f"\n{Colors.CYAN}Bilgi: Orta Risk{Colors.END}")
            print(f"{Colors.CYAN}Arıza İhtimali: {failure_prob:.2f}%{Colors.END}")
            print(f"{Colors.CYAN}Tahmin: {prediction}{Colors.END}")
        else:
            print(f"\n{Colors.GREEN}Normal: Düşük Risk{Colors.END}")
            print(f"{Colors.GREEN}Arıza İhtimali: {failure_prob:.2f}%{Colors.END}")
            print(f"{Colors.GREEN}Tahmin: {prediction}{Colors.END}")
    
    def check_api_health(self):
        """Check if API is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"{Colors.GREEN}API servisi çalışıyor durumda{Colors.END}")
                return True
            else:
                print(f"{Colors.RED}API servisi çalışmıyor (Status: {response.status_code}){Colors.END}")
                return False
        except:
            print(f"{Colors.RED}API servisine bağlanılamıyor{Colors.END}")
            return False
    
    def run_simulation(self, duration_minutes=5, interval_seconds=1):
        """Run the simulation for specified duration"""
        if not self.check_api_health():
            print(f"{Colors.RED}Simülasyon başlatılamıyor. Lütfen API servisini çalıştırın.{Colors.END}")
            return
        
        if not self.load_data():
            return
        
        total_iterations = duration_minutes * 60
        print(f"\n{Colors.BOLD}{Colors.BLUE}Simülasyon Başlatılıyor...{Colors.END}")
        print(f"{Colors.BLUE}Süre: {duration_minutes} dakika ({total_iterations} iterasyon){Colors.END}")
        print(f"{Colors.BLUE}Aralık: {interval_seconds} saniye{Colors.END}")
        print(f"{Colors.CYAN}API URL: {self.api_url}{Colors.END}")
        print(f"{Colors.YELLOW}Simülasyonu durdurmak için Ctrl+C tuşlarına basın{Colors.END}")
        
        try:
            for i in range(total_iterations):
                # Get current sensor data (cycle through the data)
                current_row = self.sensor_data.iloc[self.current_index % len(self.sensor_data)]
                self.current_index += 1
                
                # Prepare and send request
                payload = self.prepare_sensor_payload(current_row)
                result = self.send_prediction_request(payload)
                
                if result:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.print_status(timestamp, payload, result)
                
                # Wait for next iteration
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Simülasyon kullanıcı tarafından durduruldu{Colors.END}")
        except Exception as e:
            print(f"\n{Colors.RED}Simülasyon hatası: {e}{Colors.END}")
        
        print(f"{Colors.GREEN}Simülasyon tamamlandı{Colors.END}")

def main():
    """Main function to run the simulator"""
    print(f"{Colors.BOLD}{Colors.MAGENTA}Predictive Maintenance Simulator{Colors.END}")
    print(f"{Colors.CYAN}{'='*50}{Colors.END}")
    
    simulator = PredictiveMaintenanceSimulator()
    
    # Run simulation for 5 minutes by default
    simulator.run_simulation(duration_minutes=5, interval_seconds=1)

if __name__ == "__main__":
    main()
