import { Button, Card, CircularProgress, Dialog, DialogContent, DialogTitle, Tooltip, Typography } from "@mui/material";
import { motion } from "framer-motion";
import L from "leaflet";
import "leaflet.heat"; // Import leaflet.heat
import "leaflet/dist/leaflet.css";
import { useEffect, useState } from "react";
import { FaHistory, FaImage, FaMapMarkerAlt, FaSearch, FaTimes, FaUpload } from "react-icons/fa";
import { MapContainer, Marker, TileLayer, useMap, useMapEvents } from "react-leaflet";
import { useAuth } from "../context/AuthContext";

// Fix Leaflet marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png",
});

const HeatmapLayer = ({ data }) => {
  const map = useMap();

  useEffect(() => {
    if (data.length === 0) return;

    // Create heatmap layer
    const heatPoints = data.map((point) => [
      point.latitude,
      point.longitude,
      point.count // Intensity based on count
    ]);

    const heatLayer = L.heatLayer(heatPoints, {
      radius: 20,
      blur: 15,
      maxZoom: 17,
      minOpacity: 0.3,
      max: Math.max(...data.map((d) => d.count), 1),
    }).addTo(map);

    // Cleanup on unmount
    return () => {
      map.removeLayer(heatLayer);
    };
  }, [data, map]);

  return null;
};

const UploadPage = () => {
  const { isLoggedIn } = useAuth();
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [position, setPosition] = useState([12.9716, 77.5946]); // Default: Bangalore
  const [address, setAddress] = useState("");
  const [addressLoading, setAddressLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadHistory, setUploadHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState(null);
  const [heatmapData, setHeatmapData] = useState([]);

  useEffect(() => {
    if (!isLoggedIn) {
      setError("Please log in to upload water issue reports.");
    }
    const savedHistory = JSON.parse(localStorage.getItem("uploadHistory")) || [];
    setUploadHistory(savedHistory);
    fetchHeatmapData();
    // Cleanup preview URL on component unmount
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [isLoggedIn, previewUrl]);

  const fetchHeatmapData = async () => {
    try {
      const response = await fetch("http://localhost:5000/map_data");
      if (!response.ok) throw new Error("Failed to fetch heatmap data");
      const data = await response.json();
      setHeatmapData(data);
    } catch (err) {
      console.error("Error fetching heatmap data:", err);
      setError("Could not load heatmap data. Please try again later.");
    }
  };

  function LocationMarker() {
    useMapEvents({
      click(e) {
        setPosition([e.latlng.lat, e.latlng.lng]);
        fetchAddress(e.latlng.lat, e.latlng.lng);
      },
    });
    return <Marker position={position} />;
  }

  const fetchAddress = async (lat, lon) => {
    setAddressLoading(true);
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`
      );
      const data = await response.json();
      setAddress(data.display_name || "Address not found");
    } catch (error) {
      console.error("Error fetching address:", error);
      setAddress("Failed to fetch address");
      setError("Could not fetch address. Please try again.");
    } finally {
      setAddressLoading(false);
    }
  };

  const fetchCoordinates = async () => {
    if (!searchQuery) return;
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${searchQuery}`
      );
      const data = await response.json();
      if (data.length > 0) {
        const { lat, lon, display_name } = data[0];
        setPosition([parseFloat(lat), parseFloat(lon)]);
        setAddress(display_name);
      } else {
        setError("Location not found. Please try a different search term.");
      }
    } catch (error) {
      console.error("Error fetching location:", error);
      setError("Failed to fetch coordinates. Please try again.");
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith("image/")) {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setError(null);
    } else {
      setError("Please select a valid image file (e.g., JPG, PNG).");
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setError(null);
    } else {
      setError("Please drop a valid image file (e.g., JPG, PNG).");
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const handleUpload = async () => {
    if (!isLoggedIn) {
      setError("Please log in to submit a report.");
      return;
    }
    if (!selectedFile) {
      setError("Please select an image to upload.");
      return;
    }
    if (!address || address === "Failed to fetch address") {
      setError("Please select a valid location on the map.");
      return;
    }

    setUploading(true);
    setError(null);
    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("latitude", position[0]);
    formData.append("longitude", position[1]);
    formData.append("address", address);

    try {
      const token = localStorage.getItem("token");
      if (!token) {
        throw new Error("Authentication token not found. Please log in again.");
      }
      const response = await fetch("http://localhost:5000/predict_water_issue", {
        method: "POST",
        headers: {
          "x-access-token": token,
        },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.error || "Failed to submit report. Please try again."
        );
      }

      const data = await response.json();
      setPrediction(data);

      const smsSent = data.risk_score && data.risk_score > 0.7 && data.valid;
      const newEntry = {
        image: previewUrl,
        address,
        prediction: data.prediction,
        confidence: data.confidence,
        risk_score: data.risk_score,
        sms_sent: smsSent,
        timestamp: new Date().toLocaleString(),
        assignedOfficer: data.assigned_officer,
        valid: data.valid,
      };

      const updatedHistory = [newEntry, ...uploadHistory.slice(0, 4)];
      setUploadHistory(updatedHistory);
      localStorage.setItem("uploadHistory", JSON.stringify(updatedHistory));

      // Refresh heatmap data after upload
      fetchHeatmapData();

      // Clear preview after successful upload
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setSelectedFile(null);
      setPreviewUrl(null);
    } catch (error) {
      console.error("Error uploading image:", error);
      setError(
        error.message.includes("Predictive model not available")
          ? "Prediction model is currently unavailable. Report saved, but risk assessment pending."
          : error.message || "Server error. Please try again."
      );
    } finally {
      setUploading(false);
    }
  };

  const clearPreview = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-300 via-blue-400 to-purple-500 overflow-hidden">
      {/* Header */}
      <motion.header
        className="w-full py-6 text-center bg-gradient-to-r from-teal-600 to-blue-700 text-white shadow-lg"
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ type: "spring", stiffness: 120 }}
      >
        <Typography variant="h3" className="font-bold tracking-wide">
          AquaAI-Report
        </Typography>
        <Typography variant="subtitle1" className="mt-1 italic">
          "Detect Water Issues with a Single Snap!"
        </Typography>
      </motion.header>

      {/* Main Content */}
      <div className="flex flex-col lg:flex-row p-6 gap-6 max-w-7xl mx-auto">
        {/* Upload Section */}
        <motion.div
          className="lg:w-1/3"
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card className="p-6 bg-white/90 backdrop-blur-md rounded-2xl shadow-xl border border-teal-200">
            <div className="space-y-6">
              {/* Upload Area with Drag-and-Drop */}
              <motion.div
                className={`relative p-6 rounded-xl border-2 border-dashed transition-all ${
                  isDragging
                    ? "bg-blue-100 border-blue-500 scale-105"
                    : "bg-gradient-to-br from-teal-50 to-blue-50 border-teal-400"
                }`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <input
                  type="file"
                  onChange={handleFileChange}
                  className="absolute inset-0 opacity-0 cursor-pointer"
                  accept="image/*"
                />
                <FaImage
                  className={`text-4xl mx-auto mb-3 ${
                    isDragging ? "text-blue-500 animate-pulse" : "text-teal-500 animate-bounce"
                  }`}
                />
                <Typography className="text-center text-gray-700 font-medium">
                  {isDragging
                    ? "Drop it Now!"
                    : selectedFile
                    ? selectedFile.name
                    : "Drop or Select Image"}
                </Typography>
              </motion.div>

              {/* Preview */}
              {previewUrl && (
                <motion.div
                  className="relative rounded-lg overflow-hidden shadow-md"
                  initial={{ scale: 0.8 }}
                  animate={{ scale: 1 }}
                >
                  <img src={previewUrl} alt="Preview" className="w-full h-48 object-cover" />
                  <Tooltip title="Remove Image">
                    <Button
                      onClick={clearPreview}
                      className="absolute top-2 right-2 bg-red-500 text-white p-2 rounded-full hover:bg-red-600"
                    >
                      <FaTimes />
                    </Button>
                  </Tooltip>
                </motion.div>
              )}

              {/* Address */}
              <div className="bg-gray-100 p-3 rounded-lg">
                <Typography className="text-teal-700 font-semibold flex items-center gap-2">
                  <FaMapMarkerAlt />
                  {addressLoading ? (
                    <CircularProgress size={16} color="inherit" />
                  ) : address ? (
                    address
                  ) : (
                    "Select a location on the map"
                  )}
                </Typography>
              </div>

              {/* Error Message */}
              {error && (
                <Typography className="text-red-500 bg-red-100 p-3 rounded-lg text-center font-medium">
                  {error}
                </Typography>
              )}

              {/* Upload Button */}
              <Button
                variant="contained"
                onClick={handleUpload}
                disabled={uploading || !isLoggedIn}
                className="w-full py-3 bg-gradient-to-r from-teal-500 to-blue-600 hover:from-teal-600 hover:to-blue-700 rounded-full text-white font-bold"
                startIcon={uploading ? <CircularProgress size={20} color="inherit" /> : <FaUpload />}
              >
                {uploading ? "Uploading..." : "Upload & Analyze"}
              </Button>

              {/* Prediction Results */}
              {prediction && (
                <motion.div
                  className={`p-4 rounded-lg ${
                    prediction.valid
                      ? "bg-green-100 border border-green-300"
                      : "bg-yellow-100 border border-yellow-300"
                  }`}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <div className="grid grid-cols-1 gap-2">
                    <Typography className="font-bold text-lg">
                      {prediction.prediction.charAt(0).toUpperCase() +
                        prediction.prediction.slice(1)}{" "}
                      Detected
                    </Typography>
                    <div className="flex justify-between">
                      <Typography className="font-medium">Confidence:</Typography>
                      <Typography>{(prediction.confidence * 100).toFixed(2)}%</Typography>
                    </div>
                    <div className="flex justify-between">
                      <Typography className="font-medium">Risk Score:</Typography>
                      <Typography>
                        {prediction.risk_score
                          ? `${(prediction.risk_score * 100).toFixed(2)}%`
                          : "Not available"}
                      </Typography>
                    </div>
                    <div className="flex justify-between">
                      <Typography className="font-medium">Status:</Typography>
                      <Typography
                        className={
                          prediction.valid
                            ? "text-green-600 font-bold"
                            : "text-yellow-600 font-bold"
                        }
                      >
                        {prediction.valid ? "Valid Report" : "Needs Verification"}
                      </Typography>
                    </div>
                    {prediction.valid && (
                      <>
                        <div className="flex justify-between">
                          <Typography className="font-medium">Assigned Officer:</Typography>
                          <Typography className="text-blue-600">
                            {prediction.assigned_officer}
                          </Typography>
                        </div>
                        {/* <div className="flex justify-between">
                          <Typography className="font-medium">SMS Sent:</Typography>
                          <Typography
                            className={
                              prediction.risk_score > 0.7
                                ? "text-green-600 font-bold"
                                : "text-gray-600"
                            }
                          >
                            {prediction.risk_score > 0.7 ? "Yes" : "No"}
                          </Typography>
                        </div> */}
                        <Typography className="mt-2 text-sm text-gray-700">
                          Your report has been submitted successfully and assigned to an officer.
                          {prediction.risk_score > 0.7 &&
                            " An SMS has been sent to the officer."}
                        </Typography>
                      </>
                    )}
                    {!prediction.valid && (
                      <Typography className="mt-2 text-sm text-yellow-700">
                        The confidence level is below threshold. Please try with a clearer image.
                      </Typography>
                    )}
                  </div>
                </motion.div>
              )}
            </div>
          </Card>
        </motion.div>

        {/* Map Section */}
        <motion.div
          className="lg:w-2/3"
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
        >
          <Card className="p-6 bg-white/90 backdrop-blur-md rounded-2xl shadow-xl border border-blue-200">
            <div className="space-y-4">
              {/* Search Bar */}
              <div className="flex gap-3">
                <input
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search for a location..."
                  className="flex-1 p-3 bg-white rounded-full border border-teal-300 focus:ring-2 focus:ring-blue-500 outline-none"
                />
                <Button
                  variant="contained"
                  onClick={fetchCoordinates}
                  className="bg-blue-500 hover:bg-blue-600 rounded-full p-3"
                >
                  <FaSearch className="text-xl" />
                </Button>
              </div>

              {/* Map */}
              <div className="relative h-[600px] rounded-xl overflow-hidden shadow-lg">
                <MapContainer center={position} zoom={13} style={{ height: "100%", width: "100%" }}>
                  <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                  <HeatmapLayer data={heatmapData} />
                  <LocationMarker />
                </MapContainer>
                <FaMapMarkerAlt className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-blue-600 text-3xl animate-pulse" />
                <div className="absolute top-4 left-4 bg-white/80 p-2 rounded-lg shadow">
                  <Typography className="text-sm">Click to Pin Location</Typography>
                </div>
                <div className="absolute top-4 right-4 bg-white/80 p-2 rounded-lg shadow">
                  <Typography className="text-sm">Heatmap: Water Issue Hotspots</Typography>
                </div>
              </div>
            </div>
          </Card>
        </motion.div>
      </div>

      {/* Floating History Button */}
      <motion.div
        className="fixed bottom-6 right-6"
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
      >
        <Button
          variant="contained"
          onClick={() => setShowHistory(true)}
          className="bg-teal-500 hover:bg-teal-600 rounded-full p-4 shadow-lg"
        >
          <FaHistory className="text-xl" />
        </Button>
      </motion.div>

      {/* History Dialog */}
      <Dialog open={showHistory} onClose={() => setShowHistory(false)} maxWidth="sm" fullWidth>
        <DialogTitle className="bg-gradient-to-r from-teal-600 to-blue-700 text-white rounded-t-lg">
          Upload History
        </DialogTitle>
        <DialogContent className="p-4">
          {uploadHistory.length === 0 ? (
            <Typography className="text-center py-6 text-gray-500">
              No uploads yet! Start reporting water issues.
            </Typography>
          ) : (
            uploadHistory.map((entry, index) => (
              <motion.div
                key={index}
                className="flex items-center gap-4 p-3 bg-white rounded-lg shadow mb-3"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <img src={entry.image} alt="" className="w-16 h-16 object-cover rounded" />
                <div className="flex-1">
                  <Typography className="text-sm font-medium">{entry.address}</Typography>
                  <Typography
                    className={`text-sm ${
                      entry.prediction === "leakage"
                        ? "text-red-500"
                        : entry.prediction === "pollution"
                        ? "text-orange-500"
                        : entry.prediction === "scarcity"
                        ? "text-yellow-500"
                        : "text-gray-500"
                    }`}
                  >
                    {entry.prediction.charAt(0).toUpperCase() + entry.prediction.slice(1)} (
                    {(entry.confidence * 100).toFixed(2)}%)
                  </Typography>
                  <Typography className="text-xs text-blue-600">
                    Officer: {entry.assignedOfficer || "Pending assignment"}
                  </Typography>
                  <Typography className="text-xs">
                    Risk Score: {entry.risk_score ? `${(entry.risk_score * 100).toFixed(2)}%` : "N/A"}
                  </Typography>
                  {/* <Typography
                    className={`text-xs ${
                      entry.sms_sent ? "text-green-600" : "text-gray-600"
                    }`}
                  >
                    SMS: {entry.sms_sent ? "Sent to officer" : "Not sent"}
                  </Typography> */}
                  <Typography
                    className={`text-xs ${
                      entry.valid ? "text-green-600" : "text-yellow-600"
                    }`}
                  >
                    Status: {entry.valid ? "Valid" : "Needs verification"}
                  </Typography>
                  <Typography className="text-xs text-gray-500">{entry.timestamp}</Typography>
                </div>
              </motion.div>
            ))
          )}
        </DialogContent>
      </Dialog>

      {/* Footer */}
      <footer className="text-center py-4 text-gray-600 bg-white/80">
        <Typography variant="body2" className="font-semibold">
          "One Snap for Cleaner Water!"
        </Typography>
      </footer>
    </div>
  );
};

export default UploadPage;