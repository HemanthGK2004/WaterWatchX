import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Typography } from "@mui/material";
import { useAuth } from "../context/AuthContext";

const AssignedReports = () => {
  const { isLoggedIn, role } = useAuth();
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!isLoggedIn || role !== "officer") {
      setError("Please log in as an officer to view assigned reports.");
      setLoading(false);
      return;
    }

    const fetchReports = async () => {
      try {
        const token = localStorage.getItem("token");
        const response = await fetch("http://localhost:5000/officer/reports", {
          method: "GET",
          headers: {
            "x-access-token": token,
          },
        });
        const data = await response.json();
        if (response.ok) {
          setReports(data);
        } else {
          setError(data.error || "Failed to fetch reports.");
        }
      } catch (err) {
        setError("Server error. Please try again later.");
        console.error("Fetch error:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchReports();
  }, [isLoggedIn, role]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-teal-300 via-blue-400 to-purple-500">
        <Typography className="text-white text-xl">Loading...</Typography>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-300 via-blue-400 to-purple-500 p-6">
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="max-w-7xl mx-auto"
      >
        <Typography variant="h3" className="text-center text-white font-bold mb-8">
          Assigned Reports
        </Typography>

        {error && (
          <Typography className="text-red-500 bg-red-100 p-3 rounded-lg mb-6 text-center font-medium">
            {error}
          </Typography>
        )}

        {reports.length === 0 && !error ? (
          <Typography className="text-gray-600 text-center text-lg">
            No reports assigned to you yet.
          </Typography>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {reports.map((report) => (
              <motion.div
                key={report._id}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
                className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow"
              >
                <img
                  src={report.image}
                  alt="Report"
                  className="w-full h-48 object-cover rounded-lg mb-4"
                />
                <Typography className="text-lg font-semibold text-teal-600">
                  {report.status.charAt(0).toUpperCase() + report.status.slice(1)}
                </Typography>
                <Typography className="text-sm text-gray-600">
                  Address: {report.address}
                </Typography>
                <Typography className="text-sm text-gray-600">
                  Confidence: {(report.confidence * 100).toFixed(2)}%
                </Typography>
                <Typography className="text-sm text-gray-600">
                  Submitted: {new Date(report.created_at).toLocaleDateString()}
                </Typography>
                <Typography className="text-sm text-gray-600">
                  User Phone: {report.user_phone}
                </Typography>
              </motion.div>
            ))}
          </div>
        )}
      </motion.div>
    </div>
  );
};

export default AssignedReports;