import { Button, CircularProgress, Typography } from "@mui/material";
import { motion } from "framer-motion";
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import React, { useEffect, useState } from 'react';
import { FaArrowLeft, FaMapMarkedAlt, FaSyncAlt, FaWater } from "react-icons/fa";
import { Circle, MapContainer, Marker, Popup, TileLayer } from 'react-leaflet';
import { Link } from "react-router-dom";

// Fix Leaflet marker icon issue
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

// Custom icons for scarcity/leakage
const scarcityIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
  iconRetinaUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

const leakageIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png',
  iconRetinaUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

const FlowDashboard = () => {
  const [dashboardData, setDashboardData] = useState({ clusters: [], points: [] });
  const [loading, setLoading] = useState(true);
  const officerToken = localStorage.getItem('token');

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/flow_dashboard', {
        headers: { 'x-access-token': officerToken },
      });
      const data = await response.json();
      setDashboardData(data);
    } catch (error) {
      console.error('Error fetching flow dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  // Calculate priority
  const getPriority = (cluster) => {
    if (cluster.scarcity_count > 3 || cluster.leakage_count > 5) return "High";
    if (cluster.scarcity_count >= 1 || cluster.leakage_count >= 2) return "Medium";
    return "Low";
  };

  // Generate additional recommendations
  const getRecommendations = (cluster) => {
    const recs = [cluster.recommendation];
    if (cluster.scarcity_count > 0) {
      recs.push("Deploy water tankers to address immediate scarcity.");
      recs.push("Install flow meters to monitor usage.");
    }
    if (cluster.leakage_count > 0) {
      recs.push("Replace aging pipes to prevent further leaks.");
      recs.push("Adjust pressure valves to optimize flow.");
    }
    return recs;
  };

  // Generate suggestions
  const getSuggestions = (cluster) => {
    const priority = getPriority(cluster);
    const sugs = [`Priority: ${priority}`];
    if (priority === "High") {
      sugs.push("Schedule immediate maintenance.");
      sugs.push("Notify local authorities for support.");
    } else if (priority === "Medium") {
      sugs.push("Plan maintenance within the week.");
      sugs.push("Monitor with ESP8266 sensors.");
    } else {
      sugs.push("Routine inspection recommended.");
    }
    return sugs;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-teal-400 via-blue-500 to-purple-600 flex items-center justify-center">
        <CircularProgress sx={{ color: "white" }} />
      </div>
    );
  }

  const totalScarcity = dashboardData.clusters.reduce((sum, c) => sum + c.scarcity_count, 0);
  const totalLeakage = dashboardData.clusters.reduce((sum, c) => sum + c.leakage_count, 0);

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-400 via-blue-500 to-purple-600 p-6 md:p-12">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="flex items-center justify-between mb-10"
      >
        <div className="flex items-center gap-4">
          <FaMapMarkedAlt className="text-white text-5xl" />
          <Typography variant="h3" className="font-bold text-white">
            Water Flow Dashboard
          </Typography>
        </div>
        <Button
          component={Link}
          to="/officer/flow-optimization"
          variant="text"
          startIcon={<FaArrowLeft />}
          sx={{
            color: "white",
            textTransform: "none",
            fontWeight: "bold",
            fontSize: "1.1rem",
            "&:hover": { color: "#e0f7fa" },
          }}
        >
          Back to Optimization
        </Button>
      </motion.div>

      {/* Summary */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2, ease: "easeOut" }}
        className="mb-8"
      >
        <Typography variant="h6" className="text-white mb-2">
          Total Clusters: <span className="font-bold text-teal-200">{dashboardData.clusters.length}</span>
        </Typography>
        <Typography variant="body1" className="text-gray-100 mb-1">
          Scarcity Issues: <span className="font-medium text-red-300">{totalScarcity}</span>
        </Typography>
        <Typography variant="body1" className="text-gray-100">
          Leakage Issues: <span className="font-medium text-blue-300">{totalLeakage}</span>
        </Typography>
      </motion.div>

      {/* Map */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.4, ease: "easeOut" }}
        className="mb-10"
      >
        <MapContainer
          center={[12.9716, 77.5946]}
          zoom={13}
          style={{ height: '500px', width: '100%', borderRadius: '12px', boxShadow: '0 4px 15px rgba(0,0,0,0.3)' }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          {dashboardData.points.map((point, index) => (
            <Marker
              key={index}
              position={[point.latitude, point.longitude]}
              icon={point.status.toLowerCase().includes('scarcity') ? scarcityIcon : leakageIcon}
            >
              <Popup>
                {point.status} at {point.address}
              </Popup>
            </Marker>
          ))}
          {dashboardData.clusters.map((cluster) => (
            <Circle
              key={cluster.cluster_id}
              center={[cluster.center.latitude, cluster.center.longitude]}
              radius={500}
              pathOptions={{
                color: cluster.scarcity_count > cluster.leakage_count ? 'red' : 'blue',
                fillOpacity: 0.3,
                dashArray: '5, 10',
                weight: 2,
              }}
            >
              <Popup>
                <b>Cluster {cluster.cluster_id}</b><br />
                Address: {cluster.address}<br />
                Scarcity: {cluster.scarcity_count}<br />
                Leakage: {cluster.leakage_count}<br />
                Recommendation: {cluster.recommendation}
              </Popup>
            </Circle>
          ))}
        </MapContainer>
      </motion.div>

      {/* Recommendations */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6, ease: "easeOut" }}
      >
        <Typography variant="h4" className="font-bold text-white mb-6">
          Recommendations & Suggestions
        </Typography>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {dashboardData.clusters.map((cluster) => (
            <motion.div
              key={cluster.cluster_id}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, delay: 0.1 * cluster.cluster_id, ease: "easeOut" }}
              className="bg-gray-800/80 backdrop-blur-md rounded-xl p-8 border border-gray-700/50 shadow-lg hover:shadow-xl transition-shadow hover:scale-105"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className={`p-3 rounded-full ${cluster.scarcity_count > cluster.leakage_count ? "bg-red-500" : "bg-blue-500"}`}>
                  <FaWater className="text-white text-xl" />
                </div>
                <Typography variant="h5" className="font-semibold text-white">
                  Cluster {cluster.cluster_id}
                </Typography>
              </div>
              <Typography variant="body1" className="text-gray-100 mb-3">
                <span className="font-medium">Location:</span>{" "}
                {cluster.address.split(", ").slice(0, 3).join(", ")}
              </Typography>
              <Typography variant="body1" className="text-teal-300 mb-2">
                <span className="font-medium">Priority:</span> {getPriority(cluster)}
              </Typography>
              <Typography variant="body1" className="text-white mb-2">
                <span className="font-medium">Recommendations:</span>
              </Typography>
              <ul className="list-disc pl-5 text-teal-100 mb-3">
                {getRecommendations(cluster).map((rec, idx) => (
                  <li key={idx}>{rec}</li>
                ))}
              </ul>
              <Typography variant="body1" className="text-white mb-2">
                <span className="font-medium">Suggestions:</span>
              </Typography>
              <ul className="list-disc pl-5 text-gray-100">
                {getSuggestions(cluster).map((sug, idx) => (
                  <li key={idx}>{sug}</li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Action Button */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8, ease: "easeOut" }}
        className="flex justify-center mt-10"
      >
        <Button
          onClick={fetchDashboardData}
          variant="contained"
          startIcon={<FaSyncAlt />}
          disabled={loading}
          sx={{
            backgroundColor: "#14b8a6",
            color: "white",
            borderRadius: 20,
            textTransform: "none",
            fontWeight: "bold",
            padding: "12px 28px",
            boxShadow: "0 4px 15px rgba(0,0,0,0.3)",
            "&:hover": {
              backgroundColor: "#0d9488",
              boxShadow: "0 6px 20px rgba(0,0,0,0.4)",
            },
            "&:disabled": {
              backgroundColor: "#b0bec5",
              color: "#e0e0e0",
            },
          }}
        >
          {loading ? "Refreshing..." : "Refresh Data"}
        </Button>
      </motion.div>
    </div>
  );
};

export default FlowDashboard;