import { Button, CircularProgress, Typography } from "@mui/material";
import { motion } from "framer-motion";
import React, { useEffect, useState } from "react";
import { FaArrowLeft, FaMapMarkedAlt, FaSyncAlt, FaWater } from "react-icons/fa";
import { Link } from "react-router-dom";

const FlowOptimization = () => {
  const [flowData, setFlowData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [optLoading, setOptLoading] = useState(false);
  const [error, setError] = useState(null);
  const token = localStorage.getItem("token");

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/flow_dashboard", {
          headers: { "x-access-token": token },
        });
        const data = await response.json();
        console.log("Fetched flow dashboard:", data);

        if (data.error || data.clusters.length === 0) {
          console.warn("No flow dashboard data, running optimization...");
          await runOptimization();
        } else {
          setFlowData(data);
        }
      } catch (err) {
        console.error("Error fetching flow data:", err);
        setError("Failed to load flow optimization data");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const runOptimization = async () => {
    setOptLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:5000/optimize_flow", {
        method: "POST",
        headers: { "x-access-token": token },
      });
      const data = await response.json();
      if (data.error) {
        alert("Error: " + data.error);
      } else {
        setFlowData({ ...data, clusters: data.recommendations });
        alert("Optimization completed successfully!");
      }
    } catch (err) {
      alert("Failed to run optimization");
    } finally {
      setOptLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-teal-400 via-blue-500 to-purple-600 flex items-center justify-center">
        <CircularProgress sx={{ color: "white" }} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-teal-400 via-blue-500 to-purple-600 flex items-center justify-center">
        <Typography variant="h6" className="text-white font-bold">
          Error: {error}
        </Typography>
      </div>
    );
  }

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
          <FaWater className="text-white text-5xl" />
          <Typography
            variant="h3"
            className="font-bold text-white"
          >
            Water Flow Optimization
          </Typography>
        </div>
        <Button
          component={Link}
          to="/officer/profile"
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
          Back to Profile
        </Button>
      </motion.div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto">
        {flowData && flowData.clusters.length > 0 ? (
          <>
            {/* Timestamp and Summary */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2, ease: "easeOut" }}
              className="mb-8"
            >
              <Typography variant="h6" className="text-white mb-2">
                Latest optimization includes{" "}
                <span className="font-bold text-teal-200">{flowData.clusters.length}</span>{" "}
                clusters with recommendations.
              </Typography>
              <Typography variant="body1" className="text-gray-100">
                Last optimized: {new Date(flowData.created_at).toLocaleString()}
              </Typography>
            </motion.div>

            {/* Recommendations Grid */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4, ease: "easeOut" }}
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
            >
              {flowData.clusters.map((cluster) => (
                <motion.div
                  key={cluster.cluster_id}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3, delay: 0.1 * cluster.cluster_id, ease: "easeOut" }}
                  className="bg-gray-800/80 backdrop-blur-md rounded-xl p-8 border border-gray-700/50 shadow-lg hover:shadow-xl transition-shadow"
                >
                  <div className="flex items-center gap-3 mb-4">
                    <div className={`p-3 rounded-full ${cluster.recommendation.includes("scarcity") ? "bg-red-500" : "bg-blue-500"}`}>
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
                  <Typography variant="body1" className="text-teal-300">
                    <span className="font-medium">Action:</span> {cluster.recommendation}
                  </Typography>
                </motion.div>
              ))}
            </motion.div>

            {/* Action Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6, ease: "easeOut" }}
              className="flex flex-col sm:flex-row gap-4 justify-center mt-10"
            >
              <Button
                component={Link}
                to="/flow-dashboard"
                variant="contained"
                startIcon={<FaMapMarkedAlt />}
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
                }}
              >
                View Flow Dashboard
              </Button>
              <Button
                onClick={runOptimization}
                variant="contained"
                startIcon={<FaSyncAlt />}
                disabled={optLoading}
                sx={{
                  backgroundColor: "white",
                  color: "#14b8a6",
                  border: "2px solid #14b8a6",
                  borderRadius: 20,
                  textTransform: "none",
                  fontWeight: "bold",
                  padding: "12px 28px",
                  boxShadow: "0 4px 15px rgba(0,0,0,0.3)",
                  "&:hover": {
                    backgroundColor: "#e6fffa",
                    boxShadow: "0 6px 20px rgba(0,0,0,0.4)",
                  },
                  "&:disabled": {
                    backgroundColor: "#b0bec5",
                    color: "#e0e0e0",
                    borderColor: "#b0bec5",
                  },
                }}
              >
                {optLoading ? "Optimizing..." : "Run Optimization"}
              </Button>
            </motion.div>
          </>
        ) : (
          <>
            {/* No Data State */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2, ease: "easeOut" }}
              className="text-center"
            >
              <Typography variant="h5" className="text-white mb-4">
                No flow optimization data available
              </Typography>
              <Typography variant="body1" className="text-gray-100 mb-6">
                Click below to generate new recommendations based on recent reports.
              </Typography>
              <Button
                onClick={runOptimization}
                variant="contained"
                startIcon={<FaSyncAlt />}
                disabled={optLoading}
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
                {optLoading ? "Optimizing..." : "Run Optimization"}
              </Button>
            </motion.div>
          </>
        )}
      </div>
    </div>
  );
};

export default FlowOptimization;