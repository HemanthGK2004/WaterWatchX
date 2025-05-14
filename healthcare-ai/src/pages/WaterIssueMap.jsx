import React, { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { motion } from "framer-motion";
import { FaMapMarkedAlt, FaExclamationTriangle, FaCheckCircle } from "react-icons/fa";
import axios from "axios";

const WaterIssueMap = () => {
  const [reports, setReports] = useState([]);

  useEffect(() => {
    // Fetch reports from Flask backend
    axios.get("http://localhost:5000/get_reports")
      .then(response => {
        console.log("Fetched reports:", response.data);
        setReports(response.data);
      })
      .catch(error => {
        console.error("Error fetching reports:", error);
      });
  }, []);

  // Custom Icons
  const reportedIcon = new L.Icon({
    iconUrl: "https://cdn-icons-png.flaticon.com/512/684/684908.png", // Red pin for unresolved
    iconSize: [38, 38],
    iconAnchor: [19, 38],
    popupAnchor: [0, -38],
  });

  const resolvedIcon = new L.Icon({
    iconUrl: "https://cdn-icons-png.flaticon.com/512/845/845646.png", // Green check for resolved
    iconSize: [38, 38],
    iconAnchor: [19, 38],
    popupAnchor: [0, -38],
  });

  return (
    <div className="relative min-h-screen bg-gradient-to-br from-teal-100 via-blue-200 to-purple-200 p-6">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, type: "spring" }}
        className="text-center mb-6"
      >
        <div className="flex items-center justify-center gap-3">
          <FaMapMarkedAlt className="text-teal-500 text-4xl" />
          <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-teal-500 to-purple-600">
            AI-Based Water Issue Map
          </h1>
        </div>
        <p className="text-gray-700 mt-2 italic">
          “Tracking Water Issues with AI for a Sustainable Future!”
        </p>
      </motion.header>

      {/* Map Container */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6 }}
        className="relative h-[600px] w-full rounded-xl shadow-2xl overflow-hidden border border-teal-200"
      >
        <MapContainer
          center={[12.9716, 77.5946]} // Default center (Bangalore, adjustable)
          zoom={12}
          style={{ height: "100%", width: "100%" }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          />

          {reports.map((report, index) => (
            <Marker
              key={index}
              position={[report.latitude, report.longitude]}
              icon={report.resolved ? resolvedIcon : reportedIcon}
            >
              <Popup>
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.3 }}
                  className="p-3 bg-white/95 backdrop-blur-md rounded-lg shadow-md max-w-xs"
                >
                  {/* Popup Header */}
                  <div className="flex items-center gap-2 mb-2">
                    {report.resolved ? (
                      <FaCheckCircle className="text-green-500 text-xl" />
                    ) : (
                      <FaExclamationTriangle className="text-red-500 text-xl" />
                    )}
                    <h2 className="text-lg font-bold text-gray-800 truncate">
                      {report.address || "Unknown Location"}
                    </h2>
                  </div>

                  {/* Details */}
                  <div className="space-y-1 text-gray-700">
                    <p>
                      <span className="font-medium">Officer:</span>{" "}
                      {report.assigned_officer || "Not Assigned"}
                    </p>
                    <p>
                      <span className="font-medium">Issue:</span>{" "}
                      {report.status || "Unknown"}
                    </p>
                    <p>
                      <span className="font-medium">Status:</span>{" "}
                      <span
                        className={
                          report.resolved ? "text-green-600" : "text-red-600"
                        }
                      >
                        {report.resolved ? "Resolved" : "Pending"}
                      </span>
                    </p>
                  </div>

                  {/* Image */}
                  <div className="mt-3">
                    {report.resolved ? (
                      <img
                        src={report.resolved_image || "https://via.placeholder.com/150?text=Resolved+Image+Missing"}
                        alt="Resolved Water Issue"
                        className="w-48 h-48 object-cover rounded-md shadow-sm transition-transform hover:scale-105"
                        onError={(e) => (e.target.src = "https://via.placeholder.com/150?text=Image+Load+Failed")}
                      />
                    ) : (
                      <img
                        src={report.image || "https://via.placeholder.com/150?text=Water+Issue+Image+Missing"}
                        alt="Reported Water Issue"
                        className="w-48 h-48 object-cover rounded-md shadow-sm transition-transform hover:scale-105"
                        onError={(e) => (e.target.src = "https://via.placeholder.com/150?text=Image+Load+Failed")}
                      />
                    )}
                  </div>
                </motion.div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>

        {/* Map Overlay Legend */}
        <div className="absolute bottom-4 left-4 bg-white/80 backdrop-blur-md p-3 rounded-lg shadow-md">
          <div className="flex items-center gap-2">
            <img src={reportedIcon.options.iconUrl} alt="Unresolved" className="w-6 h-6" />
            <span className="text-sm text-gray-700">Unresolved</span>
          </div>
          <div className="flex items-center gap-2 mt-2">
            <img src={resolvedIcon.options.iconUrl} alt="Resolved" className="w-6 h-6" />
            <span className="text-sm text-gray-700">Resolved</span>
          </div>
        </div>
      </motion.div>

      {/* Footer */}
      <footer className="mt-6 text-center text-gray-600">
        <p className="font-semibold">“AI-Powered Water Issue Monitoring”</p>
      </footer>
    </div>
  );
};

export default WaterIssueMap;