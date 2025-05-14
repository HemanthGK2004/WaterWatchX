import axios from 'axios';
import { BarElement, CategoryScale, Chart as ChartJS, Legend, LinearScale, Title, Tooltip } from 'chart.js';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import React, { useState } from 'react';
import { Bar } from 'react-chartjs-2';
import { MapContainer, Marker, TileLayer, useMapEvents } from 'react-leaflet';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

// Fix Leaflet marker icon issue
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const MapPicker = ({ setFormData, searchedPosition }) => {
  const [position, setPosition] = useState(searchedPosition || null);

  const MapEvents = () => {
    const map = useMapEvents({
      click(e) {
        const { lat, lng } = e.latlng;
        setPosition([lat, lng]);
        setFormData((prev) => ({
          ...prev,
          latitude: lat.toFixed(6),
          longitude: lng.toFixed(6),
        }));
      },
    });
    if (searchedPosition && !position) {
      map.flyTo(searchedPosition, 12);
      setPosition(searchedPosition);
    }
    return null;
  };

  return (
    <MapContainer
      center={[12.9716, 77.5946]} // Default center: Bengaluru, India
      zoom={12}
      style={{ height: '400px', width: '100%', borderRadius: '12px', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      <MapEvents />
      {position && <Marker position={position} />}
    </MapContainer>
  );
};

const WaterQualityForm = ({ token }) => {
  const [formData, setFormData] = useState({
    ph: '',
    turbidity: '',
    temperature: '',
    conductivity: '',
    latitude: '',
    longitude: '',
  });
  const [useAutomaticParams, setUseAutomaticParams] = useState(false);
  const [useIoTSimulation, setUseIoTSimulation] = useState(false); // New state for IoT simulation
  const [iotData, setIotData] = useState(null); // Store simulated IoT data
  const [iotLoading, setIotLoading] = useState(false); // Loading state for IoT simulation
  const [result, setResult] = useState(null);
  const [insight, setInsight] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [insightLoading, setInsightLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchedAddress, setSearchedAddress] = useState('');
  const [searchedPosition, setSearchedPosition] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleToggleAutomatic = () => {
    setUseAutomaticParams(!useAutomaticParams);
    setUseIoTSimulation(false); // Disable IoT simulation if automatic is toggled
    setIotData(null); // Clear IoT data
    if (!useAutomaticParams) {
      setFormData((prev) => ({
        ...prev,
        ph: '',
        turbidity: '',
        temperature: '',
        conductivity: '',
      }));
    }
  };

  const handleToggleIoTSimulation = () => {
    setUseIoTSimulation(!useIoTSimulation);
    setUseAutomaticParams(false); // Disable automatic params if IoT is toggled
    setIotData(null); // Clear IoT data
    if (!useIoTSimulation) {
      setFormData((prev) => ({
        ...prev,
        ph: '',
        turbidity: '',
        temperature: '',
        conductivity: '',
      }));
    }
  };

  const handleSimulateIoT = async () => {
    setError(null);
    setIotData(null);
    setIotLoading(true);

    const { latitude, longitude } = formData;
    if (!latitude || !longitude) {
      setError('Please select a location on the map or search for a location');
      setIotLoading(false);
      return;
    }

    const parsedLatitude = parseFloat(latitude);
    const parsedLongitude = parseFloat(longitude);

    if (
      isNaN(parsedLatitude) ||
      parsedLatitude < -90 ||
      parsedLatitude > 90 ||
      isNaN(parsedLongitude) ||
      parsedLongitude < -180 ||
      parsedLongitude > 180
    ) {
      setError('Please enter valid coordinates: Latitude (-90 to 90), Longitude (-180 to 180)');
      setIotLoading(false);
      return;
    }

    try {
      const response = await axios.post(
        'http://localhost:5000/simulate_iot_data',
        { latitude: parsedLatitude, longitude: parsedLongitude },
        { headers: { 'x-access-token': token } }
      );
      setIotData(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred while simulating IoT data');
    } finally {
      setIotLoading(false);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery) return;
    try {
      const response = await axios.get(
        `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(searchQuery)}&format=json&limit=1`,
        { headers: { 'User-Agent': 'WaterQualityApp/1.0' } }
      );
      if (response.data.length > 0) {
        const { lat, lon, display_name } = response.data[0];
        setSearchedPosition([parseFloat(lat), parseFloat(lon)]);
        setSearchedAddress(display_name);
        setFormData((prev) => ({
          ...prev,
          latitude: parseFloat(lat).toFixed(6),
          longitude: parseFloat(lon).toFixed(6),
        }));
      } else {
        setError('Location not found');
      }
    } catch (err) {
      setError('Error searching location');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setInsight(null);
    setLoading(true);

    const { latitude, longitude } = formData;
    if (!latitude || !longitude) {
      setError('Please select a location on the map or search for a location');
      setLoading(false);
      return;
    }

    const parsedLatitude = parseFloat(latitude);
    const parsedLongitude = parseFloat(longitude);

    if (
      isNaN(parsedLatitude) ||
      parsedLatitude < -90 ||
      parsedLatitude > 90 ||
      isNaN(parsedLongitude) ||
      parsedLongitude < -180 ||
      parsedLongitude > 180
    ) {
      setError('Please enter valid coordinates: Latitude (-90 to 90), Longitude (-180 to 180)');
      setLoading(false);
      return;
    }

    let payload = {
      latitude: parsedLatitude,
      longitude: parsedLongitude,
    };

    if (useIoTSimulation && iotData) {
      payload.simulation_id = iotData.simulation_id;
    } else if (!useAutomaticParams) {
      const { ph, turbidity, temperature, conductivity } = formData;
      if (!ph || !turbidity || !temperature || !conductivity) {
        setError('All parameter fields are required for manual input');
        setLoading(false);
        return;
      }

      const parsedPh = parseFloat(ph);
      const parsedTurbidity = parseFloat(turbidity);
      const parsedTemperature = parseFloat(temperature);
      const parsedConductivity = parseFloat(conductivity);

      if (
        isNaN(parsedPh) ||
        parsedPh < 0 ||
        parsedPh > 14 ||
        isNaN(parsedTurbidity) ||
        parsedTurbidity < 0 ||
        parsedTurbidity > 100 ||
        isNaN(parsedTemperature) ||
        parsedTemperature < 0 ||
        parsedTemperature > 100 ||
        isNaN(parsedConductivity) ||
        parsedConductivity < 0 ||
        parsedConductivity > 2000
      ) {
        setError(
          'Please enter valid values: pH (0-14), Turbidity (0-100), Temperature (0-100), Conductivity (0-2000)'
        );
        setLoading(false);
        return;
      }

      payload = {
        ...payload,
        ph: parsedPh,
        turbidity: parsedTurbidity,
        temperature: parsedTemperature,
        conductivity: parsedConductivity,
      };
    }

    try {
      const response = await axios.post(
        'http://localhost:5000/predict_water_quality',
        payload,
        { headers: { 'x-access-token': token } }
      );
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred while predicting water quality');
    } finally {
      setLoading(false);
    }
  };

  const handleFetchInsight = async () => {
    if (!result?.prediction_id) return;
    setInsightLoading(true);
    setError(null);
    try {
      const response = await axios.post(
        'http://localhost:5000/water_quality_insights',
        { prediction_id: result.prediction_id },
        { headers: { 'x-access-token': token } }
      );
      setInsight(response.data.insight);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred while fetching insights');
    } finally {
      setInsightLoading(false);
    }
  };

  const chartData = result
    ? {
        labels: ['pH', 'Turbidity (NTU)', 'Temperature (°C)', 'Conductivity (µS/cm)'],
        datasets: [
          {
            label: 'Water Quality Parameters',
            data: [
              result.parameters.ph,
              result.parameters.turbidity,
              result.parameters.temperature,
              result.parameters.conductivity,
            ],
            backgroundColor: 'rgba(37, 99, 235, 0.6)',
            borderColor: 'rgba(37, 99, 235, 1)',
            borderWidth: 1,
          },
        ],
      }
    : null;

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { position: 'top', labels: { color: '#1e3a8a' } },
      title: { display: true, text: 'Water Quality Parameters', color: '#1e3a8a', font: { size: 16 } },
    },
    scales: {
      y: { beginAtZero: true, ticks: { color: '#1e3a8a' } },
      x: { ticks: { color: '#1e3a8a' } },
    },
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#e0f2fe] to-[#bfdbfe] p-6">
      <div className="max-w-full mx-auto bg-white rounded-2xl shadow-2xl p-10">
        <h2 className="text-4xl font-extrabold text-[#1e3a8a] mb-8 text-center tracking-tight">
          Water Quality Prediction
        </h2>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-[#1e3a8a] font-semibold mb-2 text-lg">Search Location</label>
            <div className="flex space-x-4">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Enter city, address, or place (e.g., Bengaluru)"
                className="flex-1 p-3 border border-[#93c5fd] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#60a5fa] transition-all duration-300 hover:border-[#60a5fa]"
              />
              <button
                type="button"
                onClick={handleSearch}
                className="bg-[#2563eb] text-[#f0f9ff] px-6 py-3 rounded-lg font-semibold hover:bg-[#1e40af] hover:shadow-lg transition-all duration-300"
              >
                Search
              </button>
            </div>
            {searchedAddress && (
              <p className="mt-2 text-[#1e3a8a] text-sm">Selected Address: {searchedAddress}</p>
            )}
          </div>
          <div>
            <label className="block text-[#1e3a8a] font-semibold mb-2 text-lg">Select Location</label>
            <MapPicker setFormData={setFormData} searchedPosition={searchedPosition} />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-[#1e3a8a] font-semibold mb-2">Latitude</label>
              <input
                type="number"
                name="latitude"
                value={formData.latitude}
                onChange={handleChange}
                placeholder="Select on map"
                className="w-full p-3 border border-[#93c5fd] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#60a5fa] transition-all duration-300 hover:border-[#60a5fa]"
                required
                step="0.0001"
                readOnly
              />
            </div>
            <div>
              <label className="block text-[#1e3a8a] font-semibold mb-2">Longitude</label>
              <input
                type="number"
                name="longitude"
                value={formData.longitude}
                onChange={handleChange}
                placeholder="Select on map"
                className="w-full p-3 border border-[#93c5fd] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#60a5fa] transition-all duration-300 hover:border-[#60a5fa]"
                required
                step="0.0001"
                readOnly
              />
            </div>
          </div>
          <div className="flex items-center space-x-6">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="useAutomaticParams"
                checked={useAutomaticParams}
                onChange={handleToggleAutomatic}
                className="mr-2 h-5 w-5 text-[#2563eb] focus:ring-[#60a5fa] border-[#93c5fd] rounded"
                disabled={useIoTSimulation}
              />
              <label htmlFor="useAutomaticParams" className="text-[#1e3a8a] font-semibold text-lg">
                Use Automatic Parameters
              </label>
            </div>
            <div className="flex items-center">
              <input
                type="checkbox"
                id="useIoTSimulation"
                checked={useIoTSimulation}
                onChange={handleToggleIoTSimulation}
                className="mr-2 h-5 w-5 text-[#2563eb] focus:ring-[#60a5fa] border-[#93c5fd] rounded"
                disabled={useAutomaticParams}
              />
              <label htmlFor="useIoTSimulation" className="text-[#1e3a8a] font-semibold text-lg">
                Use IoT Simulation
              </label>
            </div>
          </div>
          {useIoTSimulation && (
            <div className="space-y-4">
              <button
                type="button"
                onClick={handleSimulateIoT}
                disabled={iotLoading || !formData.latitude || !formData.longitude}
                className={`w-full bg-[#8b5cf6] text-[#f0f9ff] p-4 rounded-lg font-semibold text-lg transition-all duration-300 ${
                  iotLoading || !formData.latitude || !formData.longitude
                    ? 'opacity-50 cursor-not-allowed'
                    : 'hover:bg-[#7c3aed] hover:shadow-xl'
                }`}
              >
                {iotLoading ? 'Simulating IoT Data...' : 'Simulate IoT Sensor Data'}
              </button>
              {iotData && (
                <div className="p-4 bg-[#ede9fe] rounded-lg">
                  <h4 className="text-[#1e3a8a] font-semibold mb-2">Simulated IoT Sensor Data</h4>
                  <ul className="list-disc list-inside text-[#1e3a8a]">
                    <li>pH: {iotData.sensor_data.ph}</li>
                    <li>Turbidity: {iotData.sensor_data.turbidity} NTU</li>
                    <li>Temperature: {iotData.sensor_data.temperature}°C</li>
                    <li>Conductivity: {iotData.sensor_data.conductivity} µS/cm</li>
                  </ul>
                </div>
              )}
            </div>
          )}
          {!useAutomaticParams && !useIoTSimulation && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-[#1e3a8a] font-semibold mb-2">pH (0-14)</label>
                <input
                  type="number"
                  name="ph"
                  value={formData.ph}
                  onChange={handleChange}
                  placeholder="Enter pH"
                  className="w-full p-3 border border-[#93c5fd] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#60a5fa] transition-all duration-300 hover:border-[#60a5fa]"
                  required
                  step="0.1"
                />
              </div>
              <div>
                <label className="block text-[#1e3a8a] font-semibold mb-2">Turbidity (0-100 NTU)</label>
                <input
                  type="number"
                  name="turbidity"
                  value={formData.turbidity}
                  onChange={handleChange}
                  placeholder="Enter Turbidity"
                  className="w-full p-3 border border-[#93c5fd] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#60a5fa] transition-all duration-300 hover:border-[#60a5fa]"
                  required
                  step="0.1"
                />
              </div>
              <div>
                <label className="block text-[#1e3a8a] font-semibold mb-2">Temperature (0-100°C)</label>
                <input
                  type="number"
                  name="temperature"
                  value={formData.temperature}
                  onChange={handleChange}
                  placeholder="Enter Temperature"
                  className="w-full p-3 border border-[#93c5fd] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#60a5fa] transition-all duration-300 hover:border-[#60a5fa]"
                  required
                  step="0.1"
                />
              </div>
              <div>
                <label className="block text-[#1e3a8a] font-semibold mb-2">Conductivity (0-2000 µS/cm)</label>
                <input
                  type="number"
                  name="conductivity"
                  value={formData.conductivity}
                  onChange={handleChange}
                  placeholder="Enter Conductivity"
                  className="w-full p-3 border border-[#93c5fd] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#60a5fa] transition-all duration-300 hover:border-[#60a5fa]"
                  required
                  step="0.1"
                />
              </div>
            </div>
          )}
          <button
            type="submit"
            disabled={loading || (useIoTSimulation && !iotData)}
            className={`w-full bg-[#2563eb] text-[#f0f9ff] p-4 rounded-lg font-semibold text-lg transition-all duration-300 ${
              loading || (useIoTSimulation && !iotData) ? 'opacity-50 cursor-not-allowed' : 'hover:bg-[#1e40af] hover:shadow-xl'
            }`}
          >
            {loading ? 'Predicting...' : 'Predict Water Quality'}
          </button>
        </form>
        {result && (
          <div className="mt-8 p-6 bg-[#e0f2fe] rounded-xl shadow-inner">
            <h3 className="text-2xl font-semibold text-[#1e3a8a] mb-4">Prediction Result</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p><strong>Quality:</strong> {result.quality}</p>
                <p><strong>Confidence:</strong> {result.confidence}</p>
                <p><strong>Assigned Officer:</strong> {result.assigned_officer}</p>
                <p><strong>Address:</strong> {result.address}</p>
                <p><strong>Location:</strong> Lat: {result.latitude}, Lng: {result.longitude}</p>
                {result.simulation_id && <p><strong>Source:</strong> Simulated IoT Data</p>}
              </div>
              <div>
                <p><strong>Parameters:</strong></p>
                <ul className="list-disc list-inside text-[#1e3a8a]">
                  <li>pH: {result.parameters.ph}</li>
                  <li>Turbidity: {result.parameters.turbidity} NTU</li>
                  <li>Temperature: {result.parameters.temperature}°C</li>
                  <li>Conductivity: {result.parameters.conductivity} µS/cm</li>
                </ul>
              </div>
            </div>
            {chartData && (
              <div className="mt-6 bg-white p-4 rounded-lg shadow">
                <Bar data={chartData} options={chartOptions} />
              </div>
            )}
            <button
              onClick={handleFetchInsight}
              disabled={insightLoading}
              className={`mt-6 w-full bg-[#10b981] text-[#f0f9ff] p-4 rounded-lg font-semibold text-lg transition-all duration-300 ${
                insightLoading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-[#059669] hover:shadow-xl'
              }`}
            >
              {insightLoading ? 'Fetching Insight...' : 'Get AI Insight'}
            </button>
            {insight && (
              <div className="mt-4 p-4 bg-[#ecfdf5] rounded-lg">
                <p><strong>AI Insight:</strong> {insight}</p>
              </div>
            )}
          </div>
        )}
        {error && (
          <div className="mt-6 p-4 bg-red-100 text-red-700 rounded-lg">
            <p>Error: {error}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default WaterQualityForm;