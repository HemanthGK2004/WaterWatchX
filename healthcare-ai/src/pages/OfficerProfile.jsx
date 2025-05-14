import { Button, Card, CardContent, CircularProgress, Typography } from "@mui/material";
import { motion } from "framer-motion";
import React, { useEffect, useState } from "react";
import { FaCheckCircle, FaFileAlt, FaStar, FaWater } from "react-icons/fa";
import { Link } from "react-router-dom";

const OfficerProfile = () => {
  const [officer, setOfficer] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const token = localStorage.getItem("token");

  useEffect(() => {
    const fetchOfficerProfile = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/officer/profile", {
          headers: {
            "x-access-token": token,
          },
        });
        const data = await response.json();
        console.log("Fetched officer profile:", data);
        if (data.error) {
          setError(data.error);
        } else {
          setOfficer(data);
        }
      } catch (err) {
        console.error("Error fetching officer profile:", err);
        setError("Failed to load profile");
      } finally {
        setLoading(false);
      }
    };

    fetchOfficerProfile();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-teal-400 via-blue-500 to-purple-600 flex items-center justify-center">
        <CircularProgress color="inherit" />
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

  const contribution = officer.contribution;
  const isTopResolver = contribution >= 80;

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-400 via-blue-500 to-purple-600 p-8 flex items-center justify-center">
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="relative max-w-md w-full"
      >
        <Card
          className="bg-white/95 backdrop-blur-xl shadow-2xl rounded-3xl overflow-hidden"
          style={{ border: "2px solid rgba(255,255,255,0.2)" }}
        >
          <CardContent className="text-center p-8">
            <Typography
              variant="h4"
              className="font-extrabold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-teal-500 to-blue-600"
            >
              {officer.name}
            </Typography>
            {isTopResolver && (
              <motion.div
                animate={{ scale: [1, 1.1, 1] }}
                transition={{ repeat: Infinity, duration: 1.5 }}
                className="inline-block mb-4"
              >
                <div className="flex items-center gap-2 bg-yellow-100 text-yellow-600 px-3 py-1 rounded-full text-sm font-semibold shadow-md">
                  <FaStar className="text-yellow-500" />
                  Top Resolver
                </div>
              </motion.div>
            )}
            <Typography variant="h6" className="text-gray-600 mb-6 italic">
              Champion of Clean Waters!
            </Typography>

            <div className="flex justify-center mb-6">
              <motion.div
                initial={{ rotate: 0 }}
                animate={{ rotate: 360 }}
                transition={{ duration: 1, delay: 0.2 }}
              >
                <CircularProgress
                  variant="determinate"
                  value={contribution}
                  size={100}
                  thickness={6}
                  sx={{
                    color: contribution >= 80 ? "#22c55e" : "#3b82f6",
                    "& .MuiCircularProgress-circle": {
                      strokeLinecap: "round",
                    },
                  }}
                />
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center">
                  <Typography variant="h6" className="font-bold text-teal-700">
                    {contribution}%
                  </Typography>
                  <Typography variant="caption" className="text-gray-500">
                    Contribution
                  </Typography>
                </div>
              </motion.div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-center gap-3">
                <FaFileAlt className="text-blue-500 text-2xl" />
                <Typography variant="body1" className="text-gray-700 font-medium">
                  Assigned Reports: <span className="text-teal-600">{officer.assigned_reports}</span>
                </Typography>
              </div>
              <div className="flex items-center justify-center gap-3">
                <FaCheckCircle className="text-green-500 text-2xl" />
                <Typography variant="body1" className="text-gray-700 font-medium">
                  Resolved Reports: <span className="text-teal-600">{officer.resolved_reports}</span>
                </Typography>
              </div>
            </div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="mt-8 space-y-4"
            >
              <Button
                component={Link}
                to="/officer/resolved-reports"
                variant="contained"
                sx={{
                  background: "linear-gradient(to right, #14b8a6, #3b82f6)",
                  borderRadius: 20,
                  textTransform: "none",
                  fontWeight: "bold",
                  padding: "10px 24px",
                  boxShadow: "0 4px 15px rgba(0,0,0,0.3)",
                  "&:hover": {
                    background: "linear-gradient(to right, #0d9488, #2563eb)",
                    boxShadow: "0 6px 20px rgba(0,0,0,0.4)",
                  },
                }}
              >
                View Resolved Reports
              </Button>
              <Button
                component={Link}
                to="/officer/flow-optimization"
                variant="outlined"
                sx={{
                  borderColor: "#14b8a6",
                  color: "#14b8a6",
                  borderRadius: 20,
                  textTransform: "none",
                  fontWeight: "bold",
                  padding: "10px 24px",
                  "&:hover": {
                    borderColor: "#0d9488",
                    color: "#0d9488",
                    backgroundColor: "rgba(20,184,166,0.1)",
                  },
                }}
              >
                <FaWater className="mr-2" />
                Flow Optimization
              </Button>
            </motion.div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
};

export default OfficerProfile;