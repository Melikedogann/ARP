import math
import random
import tkinter as tk
from tkinter import messagebox, ttk
from dataclasses import dataclass, field
import requests
import folium
from folium import plugins
import webbrowser
from typing import List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import os
from sklearn.cluster import KMeans
import numpy as np

# Genetik algoritma parametreleri
NO_GENERATIONS = 800
POPULATION_SIZE = 150
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.35
NO_OF_MUTATIONS = 7
KEEP_BEST = True

@dataclass
class ChargingStation:
    id: int
    lat: float
    lon: float
    capacity: int = 2  # Aynı anda şarj edilebilecek araç sayısı

@dataclass
class WasteCollectionPoint:
    id: int
    name: str
    lat: float
    lon: float

@dataclass
class ElectricVehicle:
    id: int
    max_range: float = 500.0  # km cinsinden maksimum menzil
    current_charge_percentage: float = 100.0  # yüzde cinsinden mevcut şarj
    charging_rate: float = 200.0  # yüzde/saat cinsinden şarj hızı

    def drive(self, distance: float):
        energy_consumed = (distance / 10) * 10  # Her 10 km'de %10 azalma
        self.current_charge_percentage -= energy_consumed
        return max(0, self.current_charge_percentage)

    def needs_charging(self):
        return self.current_charge_percentage <= 20

    def charge(self, duration: float):
        charge_amount = duration * (self.charging_rate / 60)  # duration dakika cinsinden
        self.current_charge_percentage = min(100, self.current_charge_percentage + charge_amount)

@dataclass
class Location:
    name: str
    lat: float
    lon: float

@dataclass
class Chromosome:
    stops: List[int] = field(default_factory=list)
    fitness: float = 0.0

class WasteCollectionFrame(tk.Frame):
    def __init__(self, parent, point_id, locations=None, on_delete=None):
        super().__init__(parent, bd=1, relief=tk.GROOVE, padx=5, pady=5)
        self.point_id = point_id
        self.locations = locations or []
        self.on_delete = on_delete

        # Frame başlığı ve silme butonu için üst frame
        top_frame = tk.Frame(self)
        top_frame.pack(fill='x', expand=True)
        
        tk.Label(top_frame, text=f"Atık Toplama Noktası {point_id}:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        
        if on_delete:
            delete_btn = tk.Button(top_frame, text="Sil", fg="red", command=self._on_delete)
            delete_btn.pack(side=tk.RIGHT)
        
        # Bilgi girişi için alt frame
        input_frame = tk.Frame(self)
        input_frame.pack(fill='x', expand=True, pady=(5, 0))
        
        # Atık toplama noktası için combobox
        tk.Label(input_frame, text="Konum:").grid(row=0, column=0, sticky="w", padx=2)
        self.location_var = tk.StringVar()
        self.location_combo = tk.OptionMenu(input_frame, self.location_var, *[loc.name for loc in self.locations], command=self.update_coords)
        self.location_combo.grid(row=0, column=1, sticky="ew", padx=2)
        
        # Koordinat girişleri
        tk.Label(input_frame, text="Enlem:").grid(row=0, column=2, sticky="w", padx=2)
        self.lat_entry = tk.Entry(input_frame, width=10)
        self.lat_entry.grid(row=0, column=3, sticky="ew", padx=2)
        
        tk.Label(input_frame, text="Boylam:").grid(row=0, column=4, sticky="w", padx=2)
        self.lon_entry = tk.Entry(input_frame, width=10)
        self.lon_entry.grid(row=0, column=5, sticky="ew", padx=2)
        
        # Sütun genişliklerini ayarla
        input_frame.columnconfigure(1, weight=1)

    def _on_delete(self):
        if self.on_delete:
            self.on_delete(self)

    def update_coords(self, selection):
        for loc in self.locations:
            if loc.name == selection:
                self.lat_entry.delete(0, tk.END)
                self.lat_entry.insert(0, str(loc.lat))
                self.lon_entry.delete(0, tk.END)
                self.lon_entry.insert(0, str(loc.lon))
                break

    def get_point_data(self):
        try:
            return WasteCollectionPoint(
                id=self.point_id,
                name=self.location_var.get(),
                lat=float(self.lat_entry.get()),
                lon=float(self.lon_entry.get())
            )
        except ValueError:
            raise ValueError(f"Atık toplama noktası {self.point_id} için geçersiz koordinat değerleri")

class EVRoutingSolverApp(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.root.title("Elektrikli Atık Toplama Aracı Rotalama Sistemi")
        self.root.geometry("800x600")  # Başlangıç pencere boyutu
        
        # Mahalle verilerini yükle
        self.locations = self.load_locations()
        
        # Şarj istasyonlarını otomatik yerleştir
        self.charging_stations = self.place_charging_stations()

        # Main container
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(padx=10, pady=10, fill='both', expand=True)

        # Buttons
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(fill='x', pady=5)

        tk.Button(self.button_frame, text="Atık Toplama Noktası Ekle",
                  command=self.add_collection_point,
                  bg="#4CAF50", fg="white", padx=10).pack(side='left', padx=5)
        tk.Button(self.button_frame, text="Rota Hesapla",
                  command=self.solve_routing,
                  bg="#2196F3", fg="white", padx=10).pack(side='left', padx=5)

        # Atık toplama noktaları için başlık
        tk.Label(self.main_frame, text="Atık Toplama Noktaları", 
                 font=("Arial", 12, "bold")).pack(anchor="w", pady=(10, 5))

        # Atık toplama noktaları için frame ve scrollbar container
        self.scroll_container = tk.Frame(self.main_frame)
        self.scroll_container.pack(fill='both', expand=True, pady=5)
        
        # Atık toplama noktaları için canvas ve scrollbar oluştur
        self.canvas = tk.Canvas(self.scroll_container)
        self.scrollbar = ttk.Scrollbar(self.scroll_container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Canvas içinde frame oluştur
        self.collection_points_frame = tk.Frame(self.canvas)
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.collection_points_frame, anchor="nw")
        
        # Canvas yeniden boyutlandırıldığında frame genişliğini güncelle
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        # Frame içeriği değiştiğinde scrollbar'ı güncelle
        self.collection_points_frame.bind('<Configure>', self.on_frame_configure)
        
        self.collection_point_frames = []
        
        # İlk atık toplama noktasını otomatik ekle
        self.add_collection_point()
        
        # Mouse tekerleği ile scroll desteği (platform bağımsız)
        # Windows için
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_windows)
        # macOS için
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)
        # Trackpad için
        self.canvas.bind_all("<Control-MouseWheel>", self._on_trackpad)

    def _on_mousewheel_windows(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
    
    def _on_trackpad(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta)), "units")

    def on_canvas_configure(self, event):
        # Canvas genişliğine göre içerideki frame'i ayarla
        self.canvas.itemconfig(self.canvas_frame, width=event.width)
    
    def on_frame_configure(self, event):
        # Frame boyutu değiştiğinde scrollbar'ı güncelle
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def delete_collection_point(self, point_frame):
        # Noktayı listeden ve GUI'den kaldır
        if point_frame in self.collection_point_frames:
            self.collection_point_frames.remove(point_frame)
            point_frame.destroy()
            
            # Kalan noktaların ID'lerini yeniden numaralandır
            for i, frame in enumerate(self.collection_point_frames):
                frame.point_id = i + 1
            
            # Scrollbar'ı güncelle
            self.collection_points_frame.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def add_collection_point(self):
        point_frame = WasteCollectionFrame(
            self.collection_points_frame, 
            len(self.collection_point_frames) + 1, 
            self.locations,
            on_delete=self.delete_collection_point
        )
        point_frame.pack(fill='x', pady=5, padx=5)
        self.collection_point_frames.append(point_frame)
        # Frame güncellendiğinde scrollbar'ı güncelle
        self.collection_points_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def load_locations(self):
        try:
            excel_path = 'talep_noktalari_guncellenmis.xlsx'
            if not os.path.exists(excel_path):
                messagebox.showerror("Hata", f"'{excel_path}' dosyası bulunamadı!")
                return []
                
            df = pd.read_excel(excel_path)
            
            if 'Mahalleler' not in df.columns or 'X' not in df.columns or 'Y' not in df.columns:
                messagebox.showerror("Hata", "Excel dosyasında gerekli sütunlar (Mahalleler, X, Y) bulunamadı!")
                return []
                
            locations = []
            for _, row in df.iterrows():
                if pd.notna(row['Mahalleler']) and pd.notna(row['X']) and pd.notna(row['Y']):
                    locations.append(Location(
                        name=row['Mahalleler'],
                        lon=float(row['X']),
                        lat=float(row['Y'])
                    ))
            
            if not locations:
                messagebox.showwarning("Uyarı", "Excel dosyasından hiç konum verisi yüklenemedi!")
                
            return locations
        except Exception as e:
            messagebox.showerror("Hata", f"Konum verileri yüklenirken bir hata oluştu: {str(e)}")
            return []

    def place_charging_stations(self):
        if not self.locations:
            return []

        # Konumları numpy dizisine dönüştür
        points = np.array([[loc.lat, loc.lon] for loc in self.locations])
        
        # K-means ile 3 küme oluştur
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(points)
        
        # Her kümenin merkezini şarj istasyonu olarak kullan
        charging_stations = []
        for i, center in enumerate(kmeans.cluster_centers_):
            charging_stations.append(ChargingStation(
                id=i+1,
                lat=center[0],
                lon=center[1]
            ))
        
        return charging_stations

    def get_route_with_charging(self, start: Tuple[float, float], end: Tuple[float, float],
                              vehicle: ElectricVehicle) -> List[Tuple[float, float]]:
        """Başlangıç ve bitiş noktaları arasında şarj istasyonlarını da içeren rota oluşturur"""
        route = []
        current_pos = start
        remaining_range = vehicle.current_charge_percentage

        # Rota üzerindeki en yakın şarj istasyonunu bul
        nearest_station = min(self.charging_stations,
                            key=lambda x: get_osrm_distance(current_pos[0], current_pos[1], x.lat, x.lon))

        # Şarj istasyonuna git
        station_distance = get_osrm_distance(current_pos[0], current_pos[1], nearest_station.lat, nearest_station.lon)
        route.extend(get_osrm_route_geometry(current_pos[0], current_pos[1], nearest_station.lat, nearest_station.lon))

        # Şarj istasyonunda şarj et
        vehicle.current_charge_percentage = 100.0

        # Varış noktasına git
        route.extend(get_osrm_route_geometry(nearest_station.lat, nearest_station.lon, end[0], end[1]))

        return route

    def plot_routes(self, collection_points):
        # İlk noktayı başlangıç noktası olarak al
        start_point = collection_points[0]
        m = folium.Map(location=[start_point.lat, start_point.lon], zoom_start=12)
        
        # Atık toplama noktalarını işaretle
        for point in collection_points:
            folium.Marker(
                [point.lat, point.lon],
                popup=f'Atık Toplama Noktası: {point.name}',
                icon=folium.Icon(color='blue', icon='trash', prefix='fa')
            ).add_to(m)

        # Şarj istasyonlarını işaretle
        for station in self.charging_stations:
            folium.Marker(
                [station.lat, station.lon],
                popup=f'Şarj İstasyonu {station.id}',
                icon=folium.Icon(color='green', icon='bolt', prefix='fa')
            ).add_to(m)

        # Rotayı oluştur
        vehicle = ElectricVehicle(id=1)
        route_points = []
        current_location = (start_point.lat, start_point.lon)

        for i in range(1, len(collection_points)):
            next_point = collection_points[i]
            
            # Eğer şarj gerekiyorsa, en yakın şarj istasyonuna git
            if vehicle.needs_charging():
                nearest_station = min(self.charging_stations,
                                    key=lambda x: get_osrm_distance(current_location[0], current_location[1], x.lat, x.lon))
                
                # Şarj istasyonuna git
                charging_route = get_osrm_route_geometry(current_location[0], current_location[1],
                                                       nearest_station.lat, nearest_station.lon)
                route_points.extend(charging_route)
                
                # Şarj et
                vehicle.current_charge_percentage = 100.0
                current_location = (nearest_station.lat, nearest_station.lon)

            # Bir sonraki atık toplama noktasına git
            route_to_next = get_osrm_route_geometry(current_location[0], current_location[1],
                                                  next_point.lat, next_point.lon)
            route_points.extend(route_to_next)
            vehicle.drive(get_osrm_distance(current_location[0], current_location[1],
                                          next_point.lat, next_point.lon))
            current_location = (next_point.lat, next_point.lon)

        # AntPath ile rotayı çiz
        plugins.AntPath(
            locations=route_points,
            color='red',
            weight=5,
            opacity=0.8,
            delay=1000,
            dash_array=[10, 20],
            pulse_color='#FFFFFF'
        ).add_to(m)

        # Haritayı kaydet ve göster
        m.save('waste_collection_route.html')
        webbrowser.open('waste_collection_route.html')

    def solve_routing(self):
        try:
            if not self.collection_point_frames:
                messagebox.showerror("Hata", "Lütfen en az bir atık toplama noktası ekleyin")
                return

            collection_points = []
            for frame in self.collection_point_frames:
                collection_points.append(frame.get_point_data())

            self.plot_routes(collection_points)
            messagebox.showinfo("Başarılı", "Optimum rota hesaplandı. Harita tarayıcınızda açılacak.")

        except ValueError as e:
            messagebox.showerror("Hata", f"Geçersiz giriş: {str(e)}")
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {str(e)}")

def get_bearing(point1, point2):
    """İki nokta arasındaki açıyı hesaplar"""
    lat1, lon1 = point1
    lat2, lon2 = point2

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    diff_lon = math.radians(lon2 - lon1)

    x = math.sin(diff_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diff_lon))
    initial_bearing = math.atan2(x, y)

    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def get_osrm_route_geometry(lat1, lon1, lat2, lon2, osrm_url="http://router.project-osrm.org/route/v1/driving/"):
    url = f"{osrm_url}{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    try:
        response = requests.get(url)
        data = response.json()
        geometry = data['routes'][0]['geometry']['coordinates']
        return [(lat, lon) for lon, lat in geometry]
    except Exception as e:
        print(f"OSRM request error: {e}")
        return [(lat1, lon1), (lat2, lon2)]

def get_osrm_distance(lat1, lon1, lat2, lon2, osrm_url="http://router.project-osrm.org/route/v1/driving/"):
    url = f"{osrm_url}{lon1},{lat1};{lon2},{lat2}?overview=false"
    try:
        response = requests.get(url)
        data = response.json()
        distance = data['routes'][0]['distance'] / 1000
        return distance
    except Exception as e:
        print(f"OSRM request error: {e}")
        return math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) * 111

if __name__ == "__main__":
    root = tk.Tk()
    app = EVRoutingSolverApp(root)
    root.mainloop()